"""Vector store index types."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)


class ManagedIndexQueryMode(str, Enum):
    """Vector store query mode."""

    DEFAULT = "default"
    MMR = "mmr"
