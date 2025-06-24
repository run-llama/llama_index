from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from pathlib import Path
from unittest.mock import patch


def test_class():
    names_of_base_classes = [b.__name__ for b in FastEmbedEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


@patch("llama_index.embeddings.fastembed.base.TextEmbedding")
def test_create_fastembed_embedding(mock_text_embedding):
    cache = Path("./test_cache_2")

    fastembed_embedding = FastEmbedEmbedding(
        cache_dir=str(cache),
        embed_batch_size=24,
        doc_embed_type="passage",
    )

    assert fastembed_embedding.cache_dir == str(cache)
    assert fastembed_embedding.embed_batch_size == 24
    assert fastembed_embedding.doc_embed_type == "passage"
    assert mock_text_embedding.call_args.kwargs["cache_dir"] == str(cache)
    assert mock_text_embedding.call_args.kwargs["threads"] is None
    assert mock_text_embedding.call_args.kwargs["providers"] is None
