import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.brain import needs_vision, think_and_act
from jarvis.config import Settings


def _make_settings(**kwargs) -> Settings:
    defaults = {"groq_api_key": "test-key"}
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.fixture(autouse=True)
def reset_client():
    import jarvis.brain

    jarvis.brain._client = None
    yield
    jarvis.brain._client = None


def _text_response(text: str):
    message = MagicMock()
    message.content = text
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _tool_call_response(tool_id: str, name: str, arguments: dict):
    tool_call = MagicMock()
    tool_call.id = tool_id
    tool_call.function.name = name
    tool_call.function.arguments = __import__("json").dumps(arguments)
    message = MagicMock()
    message.content = None
    message.tool_calls = [tool_call]
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestNeedsVision:
    def test_look_keyword(self):
        s = _make_settings()
        assert needs_vision("look at this document", s) is True

    def test_see_keyword(self):
        s = _make_settings()
        assert needs_vision("can you see what's on my desk", s) is True

    def test_what_is_keyword(self):
        s = _make_settings()
        assert needs_vision("what is this thing", s) is True

    def test_no_vision_keyword(self):
        s = _make_settings()
        assert needs_vision("set a timer for 5 minutes", s) is False

    def test_case_insensitive(self):
        s = _make_settings()
        assert needs_vision("LOOK at the Screen", s) is True


class TestThinkAndAct:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = [{"type": "function", "function": {"name": "run_apple_shortcut", "description": "x", "parameters": {}}}]

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=_text_response("Hello! How can I help?"))
            mock_get_client.return_value = mock_client

            result = await think_and_act("hello", None, interrupt, tools, conversation, settings)

        assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = [{"type": "function", "function": {"name": "run_apple_shortcut", "description": "x", "parameters": {}}}]

        resp1 = _tool_call_response("call_123", "run_apple_shortcut", {"shortcut_name": "What's on today?"})
        resp2 = _text_response("You have 3 meetings today.")

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(side_effect=[resp1, resp2])
            mock_get_client.return_value = mock_client

            with patch("jarvis.hands.run_shortcut", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = "Meeting 1, Meeting 2, Meeting 3"
                result = await think_and_act(
                    "what's on my calendar",
                    None,
                    interrupt,
                    tools,
                    conversation,
                    settings,
                )

        assert result == "You have 3 meetings today."
        mock_run.assert_called_once_with("What's on today?", input_text=None)

    @pytest.mark.asyncio
    async def test_interrupt_stops_tool_loop(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = []
        interrupt.set()

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            result = await think_and_act("hello", None, interrupt, tools, conversation, settings)

        assert result == ""
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_image_included_in_message(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []
        tools = []

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=_text_response("I see a laptop on the desk."))
            mock_get_client.return_value = mock_client

            result = await think_and_act(
                "look at my desk",
                "base64imgdata",
                interrupt,
                tools,
                conversation,
                settings,
            )

        assert result == "I see a laptop on the desk."
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_msg = messages[-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][1]["type"] == "image_url"
        assert user_msg["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,base64imgdata"


class TestHarnessDispatch:
    @pytest.mark.asyncio
    async def test_save_memory_dispatch(self):
        from jarvis.brain import _execute_tool

        with patch("jarvis.harness.save_memory", return_value="Saved memory 'music'") as mock_save:
            result = await _execute_tool("save_memory", {"name": "music", "content": "Prefers Spotify"})
        assert result == "Saved memory 'music'"
        mock_save.assert_called_once_with("music", "Prefers Spotify")

    @pytest.mark.asyncio
    async def test_save_skill_dispatch(self):
        from jarvis.brain import _execute_tool

        with patch("jarvis.harness.save_skill", return_value="Saved skill 'greet'") as mock_save:
            result = await _execute_tool("save_skill", {"name": "greet", "content": "Be brief"})
        assert result == "Saved skill 'greet'"
        mock_save.assert_called_once_with("greet", "Be brief")

    @pytest.mark.asyncio
    async def test_read_harness_item_dispatch(self):
        from jarvis.brain import _execute_tool

        with patch("jarvis.harness.read_item", return_value="Prefers Spotify") as mock_read:
            result = await _execute_tool("read_harness_item", {"kind": "memory", "name": "music"})
        assert result == "Prefers Spotify"
        mock_read.assert_called_once_with("memory", "music")

    @pytest.mark.asyncio
    async def test_manage_todos_dispatch(self):
        from jarvis.brain import _execute_tool

        with patch("jarvis.harness.manage_todos", new_callable=AsyncMock, return_value="Added todo: x") as mock_mt:
            result = await _execute_tool("manage_todos", {"action": "add", "item": "x"})
        assert result == "Added todo: x"
        mock_mt.assert_called_once_with("add", "x")

    @pytest.mark.asyncio
    async def test_get_system_stats_dispatch(self):
        from jarvis.brain import _execute_tool

        with patch("jarvis.hands.get_system_stats", new_callable=AsyncMock, return_value="CPU usage: 10%.") as mock_gs:
            result = await _execute_tool("get_system_stats", {})
        assert result == "CPU usage: 10%."
        mock_gs.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_tool_exception_becomes_error_result(self):
        settings = _make_settings()
        interrupt = asyncio.Event()
        conversation: list[dict] = []

        resp1 = _tool_call_response("call_err", "save_memory", {"name": "x"})  # missing "content" → KeyError
        resp2 = _text_response("Sorry, that failed.")

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(side_effect=[resp1, resp2])
            mock_get_client.return_value = mock_client

            result = await think_and_act("remember x", None, interrupt, [], conversation, settings)

        assert result == "Sorry, that failed."
        tool_result_msg = conversation[-2]
        assert tool_result_msg["role"] == "tool"
        assert tool_result_msg["content"].startswith("Error:")


class TestSystemExtra:
    @pytest.mark.asyncio
    async def test_system_extra_appended_to_system_prompt(self):
        settings = _make_settings()
        interrupt = asyncio.Event()

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=_text_response("Hi."))
            mock_get_client.return_value = mock_client

            await think_and_act(
                "hello",
                None,
                interrupt,
                [],
                [],
                settings,
                system_extra="## Your memories\n- music: Prefers Spotify",
            )

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"].endswith("## Your memories\n- music: Prefers Spotify")
        assert system_msg["content"].startswith("You are Jarvis")

    @pytest.mark.asyncio
    async def test_no_system_extra_keeps_prompt_unchanged(self):
        from jarvis.brain import SYSTEM_PROMPT

        settings = _make_settings()
        interrupt = asyncio.Event()

        with patch("jarvis.brain._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=_text_response("Hi."))
            mock_get_client.return_value = mock_client

            await think_and_act("hello", None, interrupt, [], [], settings)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        assert messages[0]["content"] == SYSTEM_PROMPT
