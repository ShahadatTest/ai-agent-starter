"""Offline stand-ins for `anthropic.Anthropic`, used by the tests and by `python -m agent_starter.evals --offline`.

They implement only what `agent.ask` touches: `client.beta.messages.tool_runner(**kwargs)`, returning an iterable
of messages (or, with `stream=True`, of message streams with `text_stream` and `get_final_message()`).

- `ScriptedClient` replays messages you hand it. Use it to test the loop plumbing.
- `ExtractiveClient` behaves like a tiny, deterministic "model": it really calls the `search_docs` tool it was given,
  then answers with the best-matching sentence of the top passages and cites their tags. It is not a language model,
  but it exercises retrieval, the tool wrappers and the citation format end to end, which is what the offline
  evaluation measures in CI. No network, no API key.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from types import SimpleNamespace

from .retrieval import tokenize

NOT_COVERED = "The documents do not cover this question."


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name: str, input: dict | None = None, id: str = "toolu_fake") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input or {}, id=id)


def message(*blocks, stop_reason: str = "end_turn", model: str = "claude-opus-5-5", usage=None) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, model=model, usage=usage)


class FakeStream:
    """Mimics the per-turn `BetaMessageStream` the tool runner yields with `stream=True`."""

    def __init__(self, final_message, chunk_size: int = 8):
        self._final = final_message
        self._chunk_size = chunk_size

    @property
    def text_stream(self) -> Iterator[str]:
        for block in self._final.content:
            if block.type == "text":
                for i in range(0, len(block.text), self._chunk_size):
                    yield block.text[i : i + self._chunk_size]

    def get_final_message(self):
        return self._final


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._tool_runner))

    def _tool_runner(self, **kwargs) -> Iterator:
        self.calls.append(kwargs)
        messages = self._run(kwargs)
        if kwargs.get("stream"):
            return (FakeStream(m) for m in messages)
        return iter(messages)

    def _run(self, kwargs: dict) -> Iterable:  # pragma: no cover - abstract
        raise NotImplementedError


class ScriptedClient(_FakeClient):
    """Replays `messages` in order, whatever the request was."""

    def __init__(self, messages: list) -> None:
        super().__init__()
        self._messages = messages

    def _run(self, kwargs: dict) -> Iterable:
        return list(self._messages)


_PASSAGE = re.compile(r"^\[(?P<ref>[^\]]+)\](?: — (?P<heading>.*?))? \(score (?P<score>-?[\d.]+)\)\n(?P<text>.*)\Z", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def parse_search_output(output: str) -> list[dict]:
    """Splits `search_docs` output back into `{"ref", "heading", "score", "text"}` dicts (empty if no match)."""
    passages = []
    for part in output.split("\n\n---\n\n"):
        match = _PASSAGE.match(part.strip())
        if match:
            passages.append({**match.groupdict(), "score": float(match["score"])})
    return passages


class ExtractiveClient(_FakeClient):
    """Searches with the real tool, then answers extractively with citations.

    Args:
        relative_cutoff: Also cite the second passage when its score is at least this fraction of the best one.
        max_citations: How many passages an answer may cite.
    """

    def __init__(self, relative_cutoff: float = 0.75, max_citations: int = 2) -> None:
        super().__init__()
        self.relative_cutoff = relative_cutoff
        self.max_citations = max_citations

    def _run(self, kwargs: dict) -> Iterable:
        tools = {tool.name: tool for tool in kwargs["tools"]}
        question = kwargs["messages"][-1]["content"]
        call = {"query": question, "top_k": 4}
        yield message(tool_use_block("search_docs", call), stop_reason="tool_use")
        passages = parse_search_output(tools["search_docs"].call(call))
        yield message(text_block(self.answer(question, passages)))

    def answer(self, question: str, passages: list[dict]) -> str:
        if not passages:
            return NOT_COVERED
        best = passages[0]["score"]
        chosen = [p for p in passages if p["score"] >= self.relative_cutoff * best][: self.max_citations]
        query_terms = set(tokenize(question))
        sentences = []
        for passage in chosen:
            candidates = [s for s in _SENTENCE_END.split(passage["text"].replace("\n", " ")) if s.strip()]
            sentence = max(candidates, key=lambda s: len(query_terms & set(tokenize(s))))
            sentences.append(f"{sentence.strip()} [{passage['ref']}]")
        return " ".join(sentences)
