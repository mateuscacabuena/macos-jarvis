import asyncio
import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, convert_to_openai_messages
from langchain_core.runnables import RunnableConfig
from langfuse import observe
from langfuse.openai import OpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from jarvis import hands, harness
from jarvis.config import Settings

SYSTEM_PROMPT = (
    "You are Jarvis, a helpful and concise macOS voice assistant. "
    "You control the user's Mac through Apple Shortcuts and system tools. "
    "Keep responses short and conversational — they will be spoken aloud "
    "by a text-to-speech engine, so use plain spoken sentences only: "
    "never markdown, bullets, asterisks, or headings. "
    "When you execute a shortcut or tool, report the result naturally."
)

MAX_CONVERSATION_MESSAGES = 20

_client: OpenAI | None = None


def _get_client(settings: Settings) -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    return _client


def needs_vision(text: str, settings: Settings) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in settings.vision_keywords)


def _window_has_image(messages: list[dict]) -> bool:
    return any(isinstance(m.get("content"), list) for m in messages)


class _Interrupted(Exception):
    pass


@observe()
async def _execute_tool(name: str, args: dict) -> str:
    if name == "run_apple_shortcut":
        return await hands.run_shortcut(args["shortcut_name"], input_text=args.get("input_text"))
    elif name == "open_item":
        return await hands.open_item(args["path_or_app"], with_app=args.get("with_app"))
    elif name == "search_files":
        return await hands.search_files(args["query"])
    elif name == "system_maintenance":
        return await hands.system_maintenance(args["action"], dry_run=args.get("dry_run", True))
    elif name == "get_system_stats":
        return await hands.get_system_stats()
    elif name == "save_memory":
        return harness.save_memory(args["name"], args["content"])
    elif name == "save_skill":
        return harness.save_skill(args["name"], args["content"])
    elif name == "read_harness_item":
        return harness.read_item(args["kind"], args["name"])
    elif name == "manage_todos":
        return await harness.manage_todos(args["action"], args.get("item"))
    return f"Unknown tool: {name}"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@observe(name="agent_node")
async def _agent_node(state: AgentState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    interrupt: asyncio.Event = cfg["interrupt"]
    if interrupt.is_set():
        raise _Interrupted()

    openai_messages = convert_to_openai_messages(state["messages"])
    kwargs = {"model": cfg["model"], "messages": openai_messages, "max_tokens": 400, **cfg["kwargs_extra"]}
    if cfg["tools"]:
        kwargs["tools"] = cfg["tools"]

    # asyncio.to_thread copies the current contextvars context into the worker
    # thread; run_in_executor does not, which would silently detach Langfuse's
    # active-span tracking and start a new disconnected trace per LLM call.
    response = await asyncio.to_thread(cfg["client"].chat.completions.create, **kwargs)
    msg = response.choices[0].message

    ai_message: dict = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        ai_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return {"messages": [ai_message]}


@observe(name="tools_node")
async def _tools_node(state: AgentState, config: RunnableConfig) -> dict:
    interrupt: asyncio.Event = config["configurable"]["interrupt"]
    if interrupt.is_set():
        raise _Interrupted()

    last = state["messages"][-1]
    tool_messages = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        print(f"  [Brain] Tool call: {name}({json.dumps(args, ensure_ascii=False)[:120]})")
        try:
            result = await _execute_tool(name, args)
        except Exception as e:
            result = f"Error: {e}"
        print(f"  [Brain] Tool result: {str(result)[:120]}")
        tool_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    return {"messages": tool_messages}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", _tools_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


app = _build_graph()


@observe()
async def think_and_act(
    text: str,
    image: str | None,
    interrupt: asyncio.Event,
    tools: list[dict],
    conversation: list[dict],
    settings: Settings,
    system_extra: str = "",
) -> str:
    if interrupt.is_set():
        return ""

    client = _get_client(settings)

    if not conversation:
        system_content = SYSTEM_PROMPT + ("\n\n" + system_extra if system_extra else "")
        conversation.append({"role": "system", "content": system_content})

    user_content: str | list[dict]
    if image:
        user_content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
        ]
    else:
        user_content = text

    conversation.append({"role": "user", "content": user_content})
    trimmed = [conversation[0]] + conversation[1:][-MAX_CONVERSATION_MESSAGES:]

    # gpt-oss-120b rejects multimodal content arrays ("content must be a string"),
    # so any window containing an image must use a vision-capable model instead.
    using_vision_model = _window_has_image(trimmed)
    model = settings.groq_vision_model if using_vision_model else settings.groq_model
    # qwen3.6's thinking mode defaults to ~2000 output tokens, which blows past
    # Groq's free-tier OTPM limit (1000); non-thinking mode gives direct answers
    # instead of a spoken-aloud chain-of-thought dump.
    kwargs_extra = {"reasoning_effort": "none"} if using_vision_model else {}

    config: RunnableConfig = {
        "configurable": {
            "client": client,
            "model": model,
            "tools": tools,
            "kwargs_extra": kwargs_extra,
            "interrupt": interrupt,
        }
    }

    try:
        final_state = await app.ainvoke({"messages": trimmed}, config=config)
    except _Interrupted:
        return ""

    new_messages = final_state["messages"][len(trimmed) :]
    for message in new_messages:
        conversation.append(convert_to_openai_messages([message])[0])

    final_message = new_messages[-1] if new_messages else None
    return (final_message.content or "") if final_message else ""
