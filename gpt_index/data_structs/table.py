"""Struct store schema."""

from dataclasses import dataclass, field
from typing import Any, Dict

from dataclasses_json import DataClassJsonMixin

from gpt_index.data_structs.data_structs import IndexStruct


@dataclass
class StructDatapoint(DataClassJsonMixin):
    """Struct outputs."""

    # map from field name to StructValue
    fields: Dict[str, Any]


@dataclass
class BaseStructTable(IndexStruct):
    """Struct outputs."""


@dataclass
class SQLStructTable(BaseStructTable):
    """SQL struct outputs."""

    context_dict: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        # TODO: consolidate with IndexStructType
        return "sql"
