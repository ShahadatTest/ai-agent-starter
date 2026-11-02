# Contributing

Thanks for helping. This is a starter that people clone and change, so the bar is: small, readable, tested offline.

## Set up

```bash
git clone https://github.com/ThalzeronLabs/ai-agent-starter && cd ai-agent-starter
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m agent_starter.evals
```

Both commands run without an API key and without network access. Please keep it that way.

## Ground rules

- **Offline tests only.** No test may call the Anthropic API, download a model or need a key. Use the fakes in
  `agent_starter.fakes` (`ScriptedClient` to replay messages, `ExtractiveClient` for end-to-end runs) and inject
  encoders into `EmbeddingIndex` instead of loading a real model.
- **No secrets** in code, tests, fixtures or CI. Credentials come from the environment (`ANTHROPIC_API_KEY`) or an
  `ant auth login` profile.
- **Don't invent SDK parameters.** Anything passed to the Anthropic or MCP SDKs must be documented in their official
  docs or visible in the installed SDK; link the source in the PR.
- **Keep the core dependency-free** beyond `anthropic`. New heavy dependencies go behind an extra in `pyproject.toml`.
- **Keep the cached prefix stable.** Don't put per-request data (dates, user names, IDs) in `SYSTEM_PROMPT` or in tool
  descriptions; it breaks prompt caching.
- If you change retrieval, chunking or the tools, run `python -m agent_starter.evals` (and ideally
  `--retriever embeddings`) and paste the summary in the PR. If the baseline moves, update the table in the README.
- Code, comments, docs and commit messages are in English.

## Adding evaluation cases

Add one JSON object per line to `evals/cases.jsonl`:

```json
{"id": "short-kebab-id", "question": "…", "expected_citations": ["file.md#2"], "expected_facts": ["exact phrase"]}
```

`expected_citations` are chunk tags (`python -c "from agent_starter import build_retriever; [print(c.ref, c.heading) for c in build_retriever('docs-example').chunks]"`
lists them). Every expected fact must appear in one of the expected chunks; `tests/test_evals.py` checks this. Use
`[]` for questions the documents do not answer.

## Pull requests

- One topic per PR, with a test for every behavior change.
- `pytest` and `python -m agent_starter.evals` pass locally; CI must be green.
- Add a line under `## [Unreleased]` in `CHANGELOG.md`.

Looking for a first task? See the
[good first issues](https://github.com/ThalzeronLabs/ai-agent-starter/labels/good%20first%20issue).
