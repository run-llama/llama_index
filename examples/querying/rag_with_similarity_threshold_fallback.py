"""
Example: RAG with similarity-threshold-based fallback.

This example shows a simple pattern for avoiding answers when retrieval
confidence is insufficient:

1) Run a dry query (response_mode="no_text") to inspect retrieved nodes without
   calling an LLM.
2) Apply SimilarityPostprocessor(similarity_cutoff=...).
3) Gate on max retrieved node score; if below the cutoff, abstain (fallback).

This file uses MockEmbedding and MockLLM to keep the example runnable without
API keys or optional dependencies. It demonstrates control flow, not answer
quality.
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document


def run_query(query: str, similarity_cutoff: float) -> None:
    documents = [
        Document(text="The author grew up in a small town and played soccer."),
        Document(
            text=(
                "The author studied computer science and later worked in industry."
            )
        ),
        Document(text="This document is about cooking recipes and travel."),
    ]

    index = VectorStoreIndex.from_documents(documents)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

    dry_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff),
        ],
        response_mode="no_text",
    )
    dry_resp = dry_engine.query(query)
    source_nodes = dry_resp.source_nodes or []

    scores: List[float] = [
        n.score for n in source_nodes if getattr(n, "score", None) is not None
    ]
    max_score: Optional[float] = max(scores) if scores else None

    print(f"\nQuery: {query!r}")
    print(f"Similarity cutoff: {similarity_cutoff}")
    print(f"Retrieved nodes after cutoff: {len(source_nodes)}")
    if max_score is None:
        print("Max similarity score: None")
    else:
        print(f"Max similarity score: {max_score:.4f}")

    if (max_score is None) or (max_score < similarity_cutoff):
        print(
            "Fallback: No sufficiently relevant context was found to answer "
            "reliably.\n"
            "Action: Ask the user to provide more details or rephrase the "
            "question."
        )
        return

    answer_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff),
        ],
    )
    resp = answer_engine.query(query)
    print("Answer:")
    print(resp)


def main() -> None:
    Settings.embed_model = MockEmbedding(embed_dim=1536)
    Settings.llm = MockLLM(max_tokens=128)

    run_query("What did the author do growing up?", similarity_cutoff=0.0)

    # Cutoff > 1.0 forces fallback for cosine-like similarity scores.
    run_query(
        "Explain the tax rules for offshore trusts.", similarity_cutoff=1.1
    )


if __name__ == "__main__":
    main()
