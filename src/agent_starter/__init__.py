"""A small, tested starter for document-grounded Claude agents (RAG + tool use)."""

__version__ = "0.1.0"

from .agent import AgentResult, ask  # noqa: E402
from .retrieval import BM25Index, Chunk, Retriever, build_retriever, chunk_document, tokenize  # noqa: E402

__all__ = [
    "AgentResult",
    "BM25Index",
    "Chunk",
    "Retriever",
    "__version__",
    "ask",
    "build_retriever",
    "chunk_document",
    "tokenize",
]
