"""Managed index types."""

from enum import Enum


class ManagedIndexQueryMode(str, Enum):
    """Managed Index query mode."""

    DEFAULT = "default"
    MMR = "mmr"
