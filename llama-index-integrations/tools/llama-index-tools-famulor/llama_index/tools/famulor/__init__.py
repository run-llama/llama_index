"""Famulor tool spec for LlamaIndex."""

from llama_index.tools.famulor.base import FamulorToolSpec
from llama_index.tools.famulor.client import FamulorAPIError

__all__ = ["FamulorAPIError", "FamulorToolSpec"]
