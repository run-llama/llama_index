"""Struct store schema."""

from dataclasses import dataclass, field
from typing import Any, Dict

from dataclasses_json import DataClassJsonMixin

from llama_index.legacy.data_structs.data_structs import IndexStruct
from llama_index.legacy.data_structs.struct_type import IndexStructType


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
    def get_type(cls) -> IndexStructType:
        """Get type."""
        # TODO: consolidate with IndexStructType
        return IndexStructType.SQL


@dataclass
class PandasStructTable(BaseStructTable):
    """Pandas struct outputs."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.PANDAS
