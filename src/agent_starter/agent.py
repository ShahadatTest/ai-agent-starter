"""The agent: Claude + the tool runner + the document tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import anthropic

from .retrieval import Retriever
from .tools import build_tools

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You answer questions using the user's documents, which you can search with tools.

- Search before answering. If the first search is not enough, search again with other words or read the document.
- Base every factual statement on the passages you found and cite them inline with their tag, like [refunds.md#2].
- If the documents do not cover the question, say so plainly instead of guessing.
- Use the calculate tool for any arithmetic.
- Answer in the language of the question, concisely."""

# Prompt caching: a top-level `cache_control` asks the API to put the cache breakpoint on the last cacheable block of
# each request and to move it forward as the tool loop grows. The prefix is `tools` -> `system` -> `messages`, so
# keep the tool list and SYSTEM_PROMPT byte-identical between requests (no timestamps, no per-user data in them).
# Prefixes shorter than the model's minimum (512 tokens on claude-opus-5-5) are simply not cached; no error.
CACHE_CONTROL = {"type": "ephemeral"}

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


@dataclass
class AgentResult:
    answer: str
    stop_reason: str | None
    tool_calls: list[str] = field(default_factory=list)
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    """Token usage summed over every model call of the run (see `USAGE_FIELDS`)."""

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


def _add_usage(total: dict[str, int], message) -> None:
    usage = getattr(message, "usage", None)
    for name in USAGE_FIELDS:
        total[name] = total.get(name, 0) + (getattr(usage, name, None) or 0)


def ask(
    question: str,
    index: Retriever,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
    max_iterations: int = 8,
    stream: bool = False,
    on_text: Callable[[str], None] | None = None,
    cache: bool = True,
) -> AgentResult:
    """Runs the agent loop until Claude answers (or `max_iterations` model calls have been made).

    With `stream=True`, every text delta is passed to `on_text` as it arrives (the CLI prints them), and the
    result is the same `AgentResult` you get without streaming. `cache=False` turns prompt caching off.
    """
    client = client or anthropic.Anthropic()
    options = {"stream": True} if stream else {}
    if cache:
        options["cache_control"] = CACHE_CONTROL
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=64000 if stream else 16000,
        system=SYSTEM_PROMPT,
        tools=build_tools(index),
        messages=[{"role": "user", "content": question}],
        output_config={"effort": effort},
        max_iterations=max_iterations,
        # If a safety classifier declines, the API retries on a fallback model inside the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        **options,
    )

    tool_calls: list[str] = []
    usage: dict[str, int] = {}
    last = None
    for item in runner:
        if stream:
            # Each turn is a message stream: read its text as it arrives, then take the accumulated message.
            for text in item.text_stream:
                if on_text is not None:
                    on_text(text)
            item = item.get_final_message()
        last = item
        _add_usage(usage, item)
        tool_calls += [block.name for block in item.content if block.type == "tool_use"]

    if last is None:
        return AgentResult(answer="", stop_reason=None, tool_calls=tool_calls, usage=usage)
    if last.stop_reason == "refusal":
        return AgentResult("The request was declined.", "refusal", tool_calls, getattr(last, "model", None), usage)
    text = "".join(block.text for block in last.content if block.type == "text").strip()
    return AgentResult(text, last.stop_reason, tool_calls, getattr(last, "model", None), usage)
