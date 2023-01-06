"""Struct store schema."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class StructDatapoint:
    """Struct outputs."""

    # map from field name to StructValue
    fields: Dict[str, Any]
    
@dataclass
class BaseStructTable:
    """Struct outputs."""

@dataclass
class SQLiteStructTable(BaseStructTable):
    """SQLite struct outputs."""

    

