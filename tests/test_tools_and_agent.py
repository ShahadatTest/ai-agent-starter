from pathlib import Path
from types import SimpleNamespace

from agent_starter.agent import SYSTEM_PROMPT, ask
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


def _message(*blocks, stop_reason="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, model=model)


def _tool_use(name):
    return SimpleNamespace(type="tool_use", name=name)


def _text(text):
    return SimpleNamespace(type="text", text=text)


class FakeClient:
    def __init__(self, messages):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._runner))
        self._messages = messages

    def _runner(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._messages)


def test_ask_runs_the_loop_and_returns_the_final_text_with_the_tools_used():
    client = FakeClient([
        _message(_tool_use("search_docs"), stop_reason="tool_use"),
        _message(_tool_use("calculate"), stop_reason="tool_use"),
        _message(_text("Pro yearly costs $60 [plans.md#2]."), _text(" That is $5/month.")),
    ])
    result = ask("How much is Pro per month on the yearly plan?", INDEX, client=client)

    assert result.answer == "Pro yearly costs $60 [plans.md#2]. That is $5/month."
    assert result.tool_calls == ["search_docs", "calculate"]
    assert not result.refused


def test_ask_sends_the_right_request():
    client = FakeClient([_message(_text("ok"))])
    ask("Hi", INDEX, client=client, effort="high", max_iterations=3)
    [call] = client.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == SYSTEM_PROMPT
    assert call["output_config"] == {"effort": "high"}
    assert call["max_iterations"] == 3
    assert call["fallbacks"] == "default"
    assert [t.name for t in call["tools"]] == ["search_docs", "read_document", "calculate"]
    assert call["messages"] == [{"role": "user", "content": "Hi"}]


def test_refusals_are_reported_not_hidden():
    client = FakeClient([_message(_text("partial"), stop_reason="refusal")])
    result = ask("...", INDEX, client=client)
    assert result.refused
    assert result.answer == "The request was declined."
