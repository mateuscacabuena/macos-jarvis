import asyncio

from jarvis import brain
from jarvis.config import Settings
from jarvis.hands import (
    build_maintenance_tool_schema,
    build_open_tool_schema,
    build_search_tool_schema,
    build_system_stats_tool_schema,
    build_tool_schema,
    discover_shortcuts,
)
from jarvis.harness import build_context, build_harness_tool_schemas, init_harness, set_available_shortcuts


async def startup(ctx: dict) -> None:
    """Load settings and build the tool set once per worker process."""
    settings = Settings()  # type: ignore[call-arg]
    shortcut_names = await discover_shortcuts()
    init_harness()
    set_available_shortcuts(shortcut_names)

    tools = [build_tool_schema(shortcut_names)] if shortcut_names else []
    tools.extend(
        [
            build_open_tool_schema(),
            build_search_tool_schema(),
            build_maintenance_tool_schema(),
            build_system_stats_tool_schema(),
        ]
    )
    tools.extend(build_harness_tool_schemas())

    ctx["settings"] = settings
    ctx["tools"] = tools
    ctx["system_extra"] = build_context()


async def process_jarvis_command(ctx: dict, *, user_prompt: str) -> str:
    """Run a single text command through the LangGraph agent and return its reply."""
    return await brain.think_and_act(
        user_prompt,
        None,
        asyncio.Event(),
        ctx["tools"],
        [],
        ctx["settings"],
        system_extra=ctx["system_extra"],
    )
