"""Command line: python -m agent_starter --docs ./docs-example "How do refunds work?" """

from __future__ import annotations

import argparse
import sys

import anthropic

from .agent import DEFAULT_MODEL, ask
from .retrieval import BM25Index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_starter", description="Ask questions about a folder of documents.")
    parser.add_argument("question", help="Your question, in any language.")
    parser.add_argument("--docs", default="docs-example", help="Folder with .md/.txt/.rst files (default: docs-example).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--show-tools", action="store_true", help="Print which tools the agent used.")
    args = parser.parse_args(argv)

    index = BM25Index.from_folder(args.docs)
    print(f"Indexed {len(index.documents)} documents into {len(index.chunks)} chunks.", file=sys.stderr)

    try:
        result = ask(args.question, index, model=args.model, effort=args.effort)
    except anthropic.AuthenticationError:
        print("No valid Anthropic credentials. Set ANTHROPIC_API_KEY or run `ant auth login`.", file=sys.stderr)
        return 2
    except anthropic.APIConnectionError:
        print("Could not reach the Anthropic API. Check your connection.", file=sys.stderr)
        return 3

    if args.show_tools:
        print(f"Tools used: {', '.join(result.tool_calls) or 'none'}", file=sys.stderr)
    print(result.answer)
    return 0 if not result.refused else 1


if __name__ == "__main__":
    sys.exit(main())
