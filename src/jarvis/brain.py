import asyncio
import functools
import json
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall

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


async def _execute_tool(name: str, args: dict) -> str:
    if name == "run_apple_shortcut":
        return await hands.run_shortcut(args["shortcut_name"], input_text=args.get("input_text"))
    elif name == "open_item":
        return await hands.open_item(args["path_or_app"], with_app=args.get("with_app"))
    elif name == "search_files":
        return await hands.search_files(args["query"])
    elif name == "system_maintenance":
        return await hands.system_maintenance(args["action"], dry_run=args.get("dry_run", True))
    elif name == "save_memory":
        return harness.save_memory(args["name"], args["content"])
    elif name == "save_skill":
        return harness.save_skill(args["name"], args["content"])
    elif name == "read_harness_item":
        return harness.read_item(args["kind"], args["name"])
    elif name == "manage_todos":
        return await harness.manage_todos(args["action"], args.get("item"))
    return f"Unknown tool: {name}"


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

    while not interrupt.is_set():
        kwargs = {"model": settings.groq_model, "messages": trimmed}
        if tools:
            kwargs["tools"] = tools

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, functools.partial(client.chat.completions.create, **kwargs))

        msg = response.choices[0].message

        if not msg.tool_calls:
            conversation.append({"role": "assistant", "content": msg.content})
            return msg.content or ""

        function_calls = [cast(ChatCompletionMessageFunctionToolCall, tc) for tc in msg.tool_calls]

        assistant_msg = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in function_calls
            ],
        }
        trimmed.append(assistant_msg)
        conversation.append(assistant_msg)

        for tc in function_calls:
            if interrupt.is_set():
                return ""

            args = json.loads(tc.function.arguments)
            print(f"  [Brain] Tool call: {tc.function.name}({json.dumps(args, ensure_ascii=False)[:120]})")
            try:
                result = await _execute_tool(tc.function.name, args)
            except Exception as e:
                result = f"Error: {e}"
            print(f"  [Brain] Tool result: {str(result)[:120]}")

            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            trimmed.append(tool_msg)
            conversation.append(tool_msg)

    return ""
