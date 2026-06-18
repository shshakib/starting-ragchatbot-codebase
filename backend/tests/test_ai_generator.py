"""Tests for AIGenerator tool-calling behavior in ai_generator.py"""

import pytest
from unittest.mock import MagicMock, patch, call
from ai_generator import AIGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_generator():
    """Return an AIGenerator with a mocked Anthropic client."""
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        gen = AIGenerator(api_key="test-key", model="claude-test-model")
        # Expose the mock client so tests can configure it
        gen._mock_client = MockAnthropic.return_value
        gen.client = gen._mock_client
    return gen


def end_turn_response(text="Hello world"):
    """Build a mock response that ends without tool use."""
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def tool_use_response(tool_name="search_course_content", tool_input=None, tool_id="tu_abc123"):
    """Build a mock response that requests a tool call."""
    if tool_input is None:
        tool_input = {"query": "what is backpropagation"}
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = tool_input
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


def make_tool_manager(tool_result="Search results here."):
    mgr = MagicMock()
    mgr.execute_tool.return_value = tool_result
    return mgr


# ---------------------------------------------------------------------------
# generate_response — basic API interaction
# ---------------------------------------------------------------------------


def test_generate_response_calls_messages_create():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response()

    gen.generate_response(query="What is AI?")

    gen.client.messages.create.assert_called_once()


def test_generate_response_sends_user_query_in_messages():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response()

    gen.generate_response(query="What is AI?")

    call_kwargs = gen.client.messages.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "user"
    assert "What is AI?" in messages[0]["content"]


def test_generate_response_no_tools_returns_text_directly():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response("Direct answer.")

    result = gen.generate_response(query="General question?")

    assert result == "Direct answer."


def test_generate_response_uses_configured_model():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response()

    gen.generate_response(query="test")

    call_kwargs = gen.client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-test-model"


def test_generate_response_sets_temperature_zero():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response()

    gen.generate_response(query="test")

    call_kwargs = gen.client.messages.create.call_args[1]
    assert call_kwargs["temperature"] == 0


# ---------------------------------------------------------------------------
# generate_response — tool injection
# ---------------------------------------------------------------------------


def test_generate_response_adds_tools_to_api_params_when_provided():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response()
    tool_defs = [{"name": "search_course_content", "description": "search", "input_schema": {}}]

    gen.generate_response(query="test", tools=tool_defs)

    call_kwargs = gen.client.messages.create.call_args[1]
    assert call_kwargs["tools"] == tool_defs


def test_generate_response_sets_tool_choice_auto_when_tools_provided():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response()

    gen.generate_response(query="test", tools=[{"name": "any"}])

    call_kwargs = gen.client.messages.create.call_args[1]
    assert call_kwargs["tool_choice"] == {"type": "auto"}


def test_generate_response_no_tools_key_when_tools_not_provided():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response()

    gen.generate_response(query="test")

    call_kwargs = gen.client.messages.create.call_args[1]
    assert "tools" not in call_kwargs


# ---------------------------------------------------------------------------
# generate_response — tool_use delegation
# ---------------------------------------------------------------------------


def test_generate_response_delegates_to_handle_tool_execution_on_tool_use():
    gen = make_generator()
    first_resp = tool_use_response()
    final_resp = end_turn_response("Final answer after tool.")
    gen.client.messages.create.side_effect = [first_resp, final_resp]
    mgr = make_tool_manager()

    result = gen.generate_response(
        query="Explain backprop",
        tools=[{"name": "search_course_content"}],
        tool_manager=mgr,
    )

    assert result == "Final answer after tool."


def test_generate_response_no_tool_manager_returns_direct_when_tool_use():
    """If stop_reason==tool_use but no tool_manager, fall through to direct text."""
    gen = make_generator()
    resp = tool_use_response()
    # content[0] has no .text; simulate by giving it a text attr
    resp.content[0].text = "fallback"
    gen.client.messages.create.return_value = resp

    # Without a tool_manager, the code hits `return response.content[0].text`
    result = gen.generate_response(query="test", tools=[{"name": "any"}])

    assert result == "fallback"


# ---------------------------------------------------------------------------
# _run_tool_loop — single tool round (existing behavior)
# ---------------------------------------------------------------------------


def test_handle_tool_execution_calls_execute_tool_with_correct_name_and_input():
    gen = make_generator()
    first_resp = tool_use_response(
        tool_name="search_course_content",
        tool_input={"query": "backprop", "course_name": "DL 101"},
    )
    final_resp = end_turn_response("Done.")
    gen.client.messages.create.side_effect = [first_resp, final_resp]
    mgr = make_tool_manager()

    gen.generate_response(
        query="Explain backprop",
        tools=[{"name": "search_course_content"}],
        tool_manager=mgr,
    )

    mgr.execute_tool.assert_called_once_with(
        "search_course_content", query="backprop", course_name="DL 101"
    )


def test_handle_tool_execution_sends_tool_result_in_second_api_call():
    gen = make_generator()
    first_resp = tool_use_response(tool_id="tu_xyz")
    final_resp = end_turn_response("Final.")
    gen.client.messages.create.side_effect = [first_resp, final_resp]
    mgr = make_tool_manager(tool_result="Relevant course content.")

    gen.generate_response(query="test", tools=[{"name": "search_course_content"}], tool_manager=mgr)

    second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]
    # The last message should be the user-role tool_result
    tool_result_msg = messages[-1]
    assert tool_result_msg["role"] == "user"
    tool_results = tool_result_msg["content"]
    assert any(
        r.get("type") == "tool_result" and r.get("tool_use_id") == "tu_xyz" for r in tool_results
    )


def test_single_tool_round_intermediate_call_includes_tools():
    """With MAX_TOOL_ROUNDS=2, round_count=0 is an intermediate round (not the last),
    so the follow-up call includes tools — Claude can decide to use another tool or respond."""
    gen = make_generator()
    tool_defs = [{"name": "search_course_content"}]
    first_resp = tool_use_response()
    final_resp = end_turn_response("Answer.")
    gen.client.messages.create.side_effect = [first_resp, final_resp]
    mgr = make_tool_manager()

    gen.generate_response(query="test", tools=tool_defs, tool_manager=mgr)

    second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
    assert "tools" in second_call_kwargs


def test_handle_tool_execution_returns_final_response_text():
    gen = make_generator()
    gen.client.messages.create.side_effect = [
        tool_use_response(),
        end_turn_response("The final synthesized answer."),
    ]
    mgr = make_tool_manager()

    result = gen.generate_response(
        query="test", tools=[{"name": "search_course_content"}], tool_manager=mgr
    )

    assert result == "The final synthesized answer."


def test_single_tool_round_makes_exactly_two_api_calls():
    gen = make_generator()
    gen.client.messages.create.side_effect = [
        tool_use_response(),
        end_turn_response("Done."),
    ]
    mgr = make_tool_manager()

    gen.generate_response(query="test", tools=[{"name": "search_course_content"}], tool_manager=mgr)

    assert gen.client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# _run_tool_loop — two sequential tool rounds (new behavior)
# ---------------------------------------------------------------------------


def test_two_sequential_tool_rounds_makes_three_api_calls():
    gen = make_generator()
    gen.client.messages.create.side_effect = [
        tool_use_response(tool_id="tu_1"),
        tool_use_response(tool_id="tu_2"),
        end_turn_response("Final answer."),
    ]
    mgr = make_tool_manager()

    result = gen.generate_response(
        query="test", tools=[{"name": "search_course_content"}], tool_manager=mgr
    )

    assert gen.client.messages.create.call_count == 3
    assert mgr.execute_tool.call_count == 2
    assert result == "Final answer."


def test_intermediate_round_call_includes_tools():
    """When MAX_TOOL_ROUNDS=2, the second API call (round_count=0, not the last round)
    includes the tools parameter so Claude can decide to call another tool."""
    gen = make_generator()
    tool_defs = [{"name": "search_course_content"}]
    gen.client.messages.create.side_effect = [
        tool_use_response(tool_id="tu_1"),
        tool_use_response(tool_id="tu_2"),
        end_turn_response("Final."),
    ]
    mgr = make_tool_manager()

    gen.generate_response(query="test", tools=tool_defs, tool_manager=mgr)

    second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
    assert "tools" in second_call_kwargs
    assert second_call_kwargs["tool_choice"] == {"type": "auto"}


def test_two_rounds_final_call_omits_tools():
    """After two tool rounds the third (forced-text) call must omit tools."""
    gen = make_generator()
    gen.client.messages.create.side_effect = [
        tool_use_response(tool_id="tu_1"),
        tool_use_response(tool_id="tu_2"),
        end_turn_response("Final."),
    ]
    mgr = make_tool_manager()

    gen.generate_response(query="test", tools=[{"name": "search_course_content"}], tool_manager=mgr)

    third_call_kwargs = gen.client.messages.create.call_args_list[2][1]
    assert "tools" not in third_call_kwargs


def test_loop_stops_at_max_tool_rounds():
    """Even if Claude keeps returning tool_use, the loop terminates at MAX_TOOL_ROUNDS."""
    gen = make_generator()
    # Four responses available — only 3 should ever be consumed (initial + 2 rounds)
    gen.client.messages.create.side_effect = [
        tool_use_response(tool_id="tu_1"),
        tool_use_response(tool_id="tu_2"),
        end_turn_response("Capped answer."),
        end_turn_response("Should never reach this."),
    ]
    mgr = make_tool_manager()

    result = gen.generate_response(
        query="test", tools=[{"name": "search_course_content"}], tool_manager=mgr
    )

    assert gen.client.messages.create.call_count == 3
    assert result == "Capped answer."


def test_messages_accumulate_across_two_rounds():
    """After two rounds the third call's messages should contain all 5 turns:
    user, assistant(round0), user(tool_result0), assistant(round1), user(tool_result1)."""
    gen = make_generator()
    gen.client.messages.create.side_effect = [
        tool_use_response(tool_id="tu_1"),
        tool_use_response(tool_id="tu_2"),
        end_turn_response("Done."),
    ]
    mgr = make_tool_manager()

    gen.generate_response(query="test", tools=[{"name": "search_course_content"}], tool_manager=mgr)

    third_call_messages = gen.client.messages.create.call_args_list[2][1]["messages"]
    assert len(third_call_messages) == 5
    assert third_call_messages[0]["role"] == "user"  # original query
    assert third_call_messages[1]["role"] == "assistant"  # tool_use round 0
    assert third_call_messages[2]["role"] == "user"  # tool_result round 0
    assert third_call_messages[3]["role"] == "assistant"  # tool_use round 1
    assert third_call_messages[4]["role"] == "user"  # tool_result round 1


def test_tool_execution_error_does_not_raise_to_caller():
    """If execute_tool raises, the error is captured as is_error tool_result
    and generate_response returns a string rather than propagating the exception."""
    gen = make_generator()
    gen.client.messages.create.side_effect = [
        tool_use_response(),
        end_turn_response("Graceful error response."),
    ]
    mgr = make_tool_manager()
    mgr.execute_tool.side_effect = Exception("DB connection failed")

    result = gen.generate_response(
        query="test", tools=[{"name": "search_course_content"}], tool_manager=mgr
    )

    assert isinstance(result, str)


def test_tool_execution_error_sends_is_error_tool_result():
    """When execute_tool raises, the second API call receives a tool_result with is_error=True."""
    gen = make_generator()
    gen.client.messages.create.side_effect = [
        tool_use_response(tool_id="tu_err"),
        end_turn_response("Handled."),
    ]
    mgr = make_tool_manager()
    mgr.execute_tool.side_effect = Exception("timeout")

    gen.generate_response(query="test", tools=[{"name": "search_course_content"}], tool_manager=mgr)

    second_call_messages = gen.client.messages.create.call_args_list[1][1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    error_result = tool_result_msg["content"][0]
    assert error_result.get("is_error") is True
    assert error_result.get("tool_use_id") == "tu_err"


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


def test_generate_response_prepends_history_to_system_prompt():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response()

    gen.generate_response(query="follow-up", conversation_history="User: hi\nAI: hello")

    call_kwargs = gen.client.messages.create.call_args[1]
    assert "User: hi\nAI: hello" in call_kwargs["system"]


def test_generate_response_no_history_uses_base_system_prompt_only():
    gen = make_generator()
    gen.client.messages.create.return_value = end_turn_response()

    gen.generate_response(query="question")

    call_kwargs = gen.client.messages.create.call_args[1]
    assert "Previous conversation:" not in call_kwargs["system"]
