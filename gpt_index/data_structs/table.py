"""Struct store schema."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StructValue:
    """Struct output (single tuple)."""
    
    field: str
    value: str
    type: Optional[str] = None


@dataclass
class StructDatapoint:
    """Struct outputs."""

    outputs: List[StructValue]
    
@dataclass
class StructTable:
    """Struct outputs."""

    outputs: List[StructValue]
    
    

