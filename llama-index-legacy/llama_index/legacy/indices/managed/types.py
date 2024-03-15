"""Vector store index types."""
from enum import Enum


class ManagedIndexQueryMode(str, Enum):
    """Vector store query mode."""

    DEFAULT = "default"
    MMR = "mmr"
