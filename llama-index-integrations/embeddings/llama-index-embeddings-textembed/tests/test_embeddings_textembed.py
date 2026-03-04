"""Test the TextEmbed class."""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.textembed import TextEmbedEmbedding


def test_textembed_class():
    """Check if BaseEmbedding is one of the base classes of TextEmbedEmbedding."""
    assert issubclass(TextEmbedEmbedding, BaseEmbedding), (
        "TextEmbedEmbedding does not inherit from BaseEmbedding"
    )


if __name__ == "__main__":
    test_textembed_class()
    print("All tests passed.")
