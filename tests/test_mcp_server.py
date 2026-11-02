"""The MCP server, exercised through a real MCP client: in-process and over stdio. No network, no key."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="install the [mcp] extra to test the MCP server")

from mcp import Client, StdioServerParameters  # noqa: E402

from agent_starter.mcp_server import create_server  # noqa: E402
from agent_starter.retrieval import BM25Index  # noqa: E402
from agent_starter.tools import build_tools  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INDEX = BM25Index.from_folder(ROOT / "docs-example")


def _run(coro):
    return asyncio.run(coro)


async def _list_tools(server):
    async with Client(server) as client:
        return (await client.list_tools()).tools


async def _call(server, name, arguments):
    async with Client(server) as client:
        result = await client.call_tool(name, arguments)
        return result.is_error, result.content[0].text


def test_exposes_the_same_tools_as_the_agent_with_the_same_descriptions():
    tools = {t.name: t for t in _run(_list_tools(create_server(INDEX)))}
    agent_tools = {t.name: t.to_dict() for t in build_tools(INDEX)}
    assert set(tools) == set(agent_tools) == {"search_docs", "read_document", "calculate"}
    for name, spec in agent_tools.items():
        assert tools[name].description == spec["description"]
        assert tools[name].input_schema["required"] == spec["input_schema"]["required"]
        for param, schema in spec["input_schema"]["properties"].items():
            assert tools[name].input_schema["properties"][param]["description"] == schema["description"]
        assert tools[name].annotations.read_only_hint is True


def test_search_docs_returns_cited_passages():
    is_error, text = _run(_call(create_server(INDEX), "search_docs", {"query": "students discount", "top_k": 2}))
    assert not is_error
    assert text.startswith("[plans.md#4]") and "50% off Pro" in text


def test_read_document_and_calculate():
    server = create_server(INDEX)
    assert "Refund window" in _run(_call(server, "read_document", {"source": "refunds.md"}))[1]
    assert "There is no document" in _run(_call(server, "read_document", {"source": "nope.md"}))[1]
    assert _run(_call(server, "calculate", {"expression": "60 / 12"}))[1] == "5"
    assert _run(_call(server, "calculate", {"expression": "__import__('os')"}))[1].startswith("Error")


def test_invalid_arguments_are_reported_as_tool_errors():
    is_error, _ = _run(_call(create_server(INDEX), "search_docs", {}))
    assert is_error


def test_stdio_entry_point_end_to_end():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_starter.mcp_server", "--docs", str(ROOT / "docs-example")],
        env=env,
    )
    is_error, text = _run(_call(params, "search_docs", {"query": "Frankfurt"}))
    assert not is_error and text.startswith("[security.md#2]")
