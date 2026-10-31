# agent-starter

**A small, tested starter for document-grounded Claude agents: retrieval + tool use + citations, in ~300 lines of Python.**

Point it at a folder of Markdown/text files and ask questions. Claude searches the documents with tools, reads more
when it needs to, does arithmetic with a safe calculator, and answers with inline citations like `[refunds.md#2]`.

```bash
pip install -e ".[dev]"
python -m agent_starter --docs docs-example --show-tools "Can I get my money back on a yearly Pro plan after 10 days?"
```

Illustrative output (the exact wording varies from run to run):

```
Tools used: search_docs, search_docs
Yes. Annual plans can be refunded in full within 14 days of purchase or renewal [refunds.md#1], and 10 days is
inside that window. Email billing@northwind-notes.example with your account email and the charge date [refunds.md#2].
```

## How it works

```
question ──▶ Claude (claude-opus-5-5, tool runner) ──▶ search_docs ──▶ BM25 index over chunked docs
                       │   ▲                          read_document ──▶ whole file
                       │   └──── tool results ◀───── calculate     ──▶ safe AST arithmetic
                       ▼
             answer with [source#chunk] citations
```

- **Agent loop:** the official Anthropic SDK's **tool runner** (`client.beta.messages.tool_runner`). Tools are plain
  Python functions; `@beta_tool` builds their JSON schemas from type hints and docstrings.
- **Retrieval:** chunks split on Markdown headings and packed by size, ranked with **Okapi BM25** (accent- and
  case-insensitive, English + Portuguese stopwords). No vector database, no embeddings API, deterministic.
  Swap in embeddings later by replacing one class.
- **Grounding:** the system prompt requires citations and an explicit "not covered" when the docs don't answer.
- **Safety:** the calculator walks the Python AST and only allows numbers and arithmetic operators; no `eval`.
- **Robustness:** refusals (`stop_reason == "refusal"`) are reported, with server-side fallbacks enabled; auth and
  connection errors get clear CLI messages.

## Tests

```bash
pytest
```

20 tests run offline (no key, no network): tokenization, chunking, BM25 ranking, every tool, the calculator's
sandbox, the generated tool schemas, and the agent loop against a scripted fake client.

## Use your own documents

```bash
python -m agent_starter --docs path/to/your/handbook "What is the vacation policy?"
```

Credentials come from `ANTHROPIC_API_KEY` or an `ant auth login` profile. `--effort low|medium|high|xhigh|max`
trades depth for cost.

## License

Apache-2.0 © Esdras Souza (Thalzeron Labs). The example documents describe a fictional company.
