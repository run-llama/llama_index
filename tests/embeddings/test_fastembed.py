from typing import Literal

import pytest
from llama_index.embeddings import FastEmbedEmbedding

try:
    import fastembed
except ImportError:
    fastembed = None  # type: ignore


@pytest.mark.skipif(fastembed is None, reason="fastembed is not installed")
@pytest.mark.parametrize(
    "model_name", ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
)
@pytest.mark.parametrize("max_length", [50, 512])
@pytest.mark.parametrize("doc_embed_type", ["default", "passage"])
@pytest.mark.parametrize("threads", [0, 10])
def test_fastembed_embedding_texts_batch(
    model_name: str,
    max_length: int,
    doc_embed_type: Literal["default", "passage"],
    threads: int,
) -> None:
    """Test FastEmbed batch embedding."""
    documents = ["foo bar", "bar foo"]
    embedding = FastEmbedEmbedding(
        model_name=model_name,
        max_length=max_length,
        doc_embed_type=doc_embed_type,
        threads=threads,
    )

    output = embedding.get_text_embedding_batch(documents)
    assert len(output) == len(documents)
    assert len(output[0]) == 384


@pytest.mark.skipif(fastembed is None, reason="fastembed is not installed")
@pytest.mark.parametrize(
    "model_name", ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
)
@pytest.mark.parametrize("max_length", [50, 512])
def test_fastembed_query_embedding(model_name: str, max_length: int) -> None:
    """Test FastEmbed batch embedding."""
    query = "foo bar"
    embedding = FastEmbedEmbedding(
        model_name=model_name,
        max_length=max_length,
    )

    output = embedding.get_query_embedding(query)
    assert len(output) == 384
