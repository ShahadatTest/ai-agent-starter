"""The agent: Claude + the tool runner + the document tools."""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from .retrieval import BM25Index
from .tools import build_tools

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You answer questions using the user's documents, which you can search with tools.

- Search before answering. If the first search is not enough, search again with other words or read the document.
- Base every factual statement on the passages you found and cite them inline with their tag, like [refunds.md#2].
- If the documents do not cover the question, say so plainly instead of guessing.
- Use the calculate tool for any arithmetic.
- Answer in the language of the question, concisely."""


@dataclass
class AgentResult:
    answer: str
    stop_reason: str | None
    tool_calls: list[str] = field(default_factory=list)
    model: str | None = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


def ask(
    question: str,
    index: BM25Index,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
    max_iterations: int = 8,
) -> AgentResult:
    """Runs the agent loop until Claude answers (or `max_iterations` model calls have been made)."""
    client = client or anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=build_tools(index),
        messages=[{"role": "user", "content": question}],
        output_config={"effort": effort},
        max_iterations=max_iterations,
        # If a safety classifier declines, the API retries on a fallback model inside the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    tool_calls: list[str] = []
    last = None
    for message in runner:
        last = message
        tool_calls += [block.name for block in message.content if block.type == "tool_use"]

    if last is None:
        return AgentResult(answer="", stop_reason=None, tool_calls=tool_calls)
    if last.stop_reason == "refusal":
        return AgentResult("The request was declined.", "refusal", tool_calls, getattr(last, "model", None))
    text = "".join(block.text for block in last.content if block.type == "text").strip()
    return AgentResult(text, last.stop_reason, tool_calls, getattr(last, "model", None))
