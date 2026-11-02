from pathlib import Path
from types import SimpleNamespace

from agent_starter.agent import SYSTEM_PROMPT, ask
from agent_starter.fakes import ScriptedClient, message, text_block, tool_use_block
from agent_starter.retrieval import BM25Index
from agent_starter.tools import build_tools, calculate_impl, read_document_impl, search_docs_impl

INDEX = BM25Index.from_folder(Path(__file__).resolve().parents[1] / "docs-example")


# --- tools -------------------------------------------------------------------------------------------------------


def test_search_returns_cited_passages():
    out = search_docs_impl(INDEX, "students discount")
    assert out.startswith("[plans.md#")
    assert "50% off Pro" in out


def test_search_says_so_when_nothing_matches():
    assert "No passage" in search_docs_impl(INDEX, "submarine")


def test_top_k_is_clamped():
    assert search_docs_impl(INDEX, "plan", top_k=999).count("[") <= 8


def test_read_document_and_unknown_names():
    assert "Refund window" in read_document_impl(INDEX, "refunds.md")
    missing = read_document_impl(INDEX, "nope.md")
    assert "There is no document" in missing and "refunds.md" in missing


def test_calculator_does_exact_arithmetic():
    assert calculate_impl("60 / 12") == "5"
    assert calculate_impl("6 * 12 - 60") == "12"
    assert calculate_impl("2 ** 10") == "1024"
    assert calculate_impl("-(3 + 4) * 2") == "-14"


def test_calculator_refuses_anything_but_arithmetic():
    assert calculate_impl("__import__('os').system('echo hi')").startswith("Error")
    assert calculate_impl("open('x')").startswith("Error")
    assert calculate_impl("9 ** 999999").startswith("Error")
    assert calculate_impl("1 / 0") == "Error: division by zero"
    assert calculate_impl("2 +").startswith("Error")


def test_tool_schemas_come_from_signatures_and_docstrings():
    tools = {t.name: t.to_dict() for t in build_tools(INDEX)}
    assert set(tools) == {"search_docs", "read_document", "calculate"}
    schema = tools["search_docs"]["input_schema"]
    assert schema["required"] == ["query"]
    assert "top_k" in schema["properties"]
    assert "Search the user's documents" in tools["search_docs"]["description"]


# --- agent loop (fake client: no network, no key) ----------------------------------------------------------------


def _usage(**counts):
    return SimpleNamespace(**counts)


def test_ask_runs_the_loop_and_returns_the_final_text_with_the_tools_used():
    client = ScriptedClient([
        message(tool_use_block("search_docs"), stop_reason="tool_use"),
        message(tool_use_block("calculate"), stop_reason="tool_use"),
        message(text_block("Pro yearly costs $60 [plans.md#2]."), text_block(" That is $5/month.")),
    ])
    result = ask("How much is Pro per month on the yearly plan?", INDEX, client=client)

    assert result.answer == "Pro yearly costs $60 [plans.md#2]. That is $5/month."
    assert result.tool_calls == ["search_docs", "calculate"]
    assert not result.refused


def test_ask_sends_the_right_request():
    client = ScriptedClient([message(text_block("ok"))])
    ask("Hi", INDEX, client=client, effort="high", max_iterations=3)
    [call] = client.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == SYSTEM_PROMPT
    assert call["output_config"] == {"effort": "high"}
    assert call["max_iterations"] == 3
    assert call["fallbacks"] == "default"
    assert [t.name for t in call["tools"]] == ["search_docs", "read_document", "calculate"]
    assert call["messages"] == [{"role": "user", "content": "Hi"}]
    assert "stream" not in call


def test_prompt_caching_is_on_by_default_and_can_be_turned_off():
    client = ScriptedClient([message(text_block("ok"))])
    ask("Hi", INDEX, client=client)
    ask("Hi", INDEX, client=client, cache=False)
    assert client.calls[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in client.calls[1]


def test_the_system_prompt_is_static_so_the_cached_prefix_stays_identical():
    client = ScriptedClient([message(text_block("ok"))])
    ask("first question", INDEX, client=client)
    ask("second question", INDEX, client=client)
    first, second = client.calls
    assert first["system"] == second["system"]
    assert [t.to_dict() for t in first["tools"]] == [t.to_dict() for t in second["tools"]]


def test_streaming_passes_text_deltas_as_they_arrive_and_returns_the_same_result():
    turns = [
        message(tool_use_block("search_docs"), stop_reason="tool_use"),
        message(text_block("Refunds are possible within 14 days [refunds.md#1]."), stop_reason="end_turn"),
    ]
    client = ScriptedClient(turns)
    deltas = []
    result = ask("Refunds?", INDEX, client=client, stream=True, on_text=deltas.append)

    assert len(deltas) > 1  # arrived in pieces
    assert "".join(deltas) == "Refunds are possible within 14 days [refunds.md#1]."
    assert result.answer == "Refunds are possible within 14 days [refunds.md#1]."
    assert result.tool_calls == ["search_docs"]
    [call] = client.calls
    assert call["stream"] is True
    assert call["max_tokens"] == 64000


def test_streaming_without_a_callback_still_returns_the_answer():
    client = ScriptedClient([message(text_block("ok"))])
    assert ask("Hi", INDEX, client=client, stream=True).answer == "ok"


def test_usage_is_summed_over_every_model_call():
    client = ScriptedClient([
        message(tool_use_block("search_docs"), stop_reason="tool_use",
                usage=_usage(input_tokens=900, output_tokens=40, cache_creation_input_tokens=800,
                             cache_read_input_tokens=0)),
        message(text_block("ok"),
                usage=_usage(input_tokens=300, output_tokens=60, cache_creation_input_tokens=0,
                             cache_read_input_tokens=800)),
    ])
    result = ask("Hi", INDEX, client=client)
    assert result.usage == {"input_tokens": 1200, "output_tokens": 100,
                            "cache_creation_input_tokens": 800, "cache_read_input_tokens": 800}


def test_refusals_are_reported_not_hidden():
    client = ScriptedClient([message(text_block("partial"), stop_reason="refusal")])
    result = ask("...", INDEX, client=client)
    assert result.refused
    assert result.answer == "The request was declined."


def test_an_empty_run_is_safe():
    result = ask("Hi", INDEX, client=ScriptedClient([]))
    assert result.answer == "" and result.stop_reason is None
