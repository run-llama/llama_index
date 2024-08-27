"""Test finetuning engine."""

import pkgutil

import pytest


def test_torch_imports() -> None:
    """Test that torch is an optional dependency."""
    # importing fine-tuning modules should be ok
    from llama_index.finetuning import EmbeddingAdapterFinetuneEngine  # noqa
    from llama_index.finetuning import OpenAIFinetuneEngine  # noqa
    from llama_index.finetuning import AzureOpenAIFinetuneEngine  # noqa
    from llama_index.finetuning import SentenceTransformersFinetuneEngine  # noqa

    # if torch isn't installed, then these should fail
    if pkgutil.find_loader("torch") is None:
        with pytest.raises(ModuleNotFoundError):
            from llama_index.embeddings.adapter.utils import LinearLayer
    else:
        # else, importing these should be ok
        from llama_index.embeddings.adapter.utils import LinearLayer  # noqa
