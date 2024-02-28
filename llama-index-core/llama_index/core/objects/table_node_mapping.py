"""Table node mapping."""

from typing import Any, Dict, Optional, Sequence

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.objects.base_node_mapping import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    BaseObjectNodeMapping,
)
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.utilities.sql_wrapper import SQLDatabase


class SQLTableSchema(BaseModel):
    """Lightweight representation of a SQL table."""

    table_name: str
    context_str: Optional[str] = None


class SQLTableNodeMapping(BaseObjectNodeMapping[SQLTableSchema]):
    """SQL Table node mapping."""

    def __init__(self, sql_database: SQLDatabase) -> None:
        self._sql_database = sql_database

    @classmethod
    def from_objects(
        cls,
        objs: Sequence[SQLTableSchema],
        *args: Any,
        sql_database: Optional[SQLDatabase] = None,
        **kwargs: Any,
    ) -> "BaseObjectNodeMapping":
        """Initialize node mapping."""
        if sql_database is None:
            raise ValueError("Must provide sql_database")
        # ignore objs, since we are building from sql_database
        return cls(sql_database)

    def _add_object(self, obj: SQLTableSchema) -> None:
        raise NotImplementedError

    def to_node(self, obj: SQLTableSchema) -> TextNode:
        """To node."""
        # taken from existing schema logic
        table_text = (
            f"Schema of table {obj.table_name}:\n"
            f"{self._sql_database.get_single_table_info(obj.table_name)}\n"
        )

        metadata = {"name": obj.table_name}

        if obj.context_str is not None:
            table_text += f"Context of table {obj.table_name}:\n"
            table_text += obj.context_str
            metadata["context"] = obj.context_str

        return TextNode(
            text=table_text,
            metadata=metadata,
            excluded_embed_metadata_keys=["name", "context"],
            excluded_llm_metadata_keys=["name", "context"],
        )

    def _from_node(self, node: BaseNode) -> SQLTableSchema:
        """From node."""
        if node.metadata is None:
            raise ValueError("Metadata must be set")
        return SQLTableSchema(
            table_name=node.metadata["name"], context_str=node.metadata.get("context")
        )

    @property
    def obj_node_mapping(self) -> Dict[int, Any]:
        """The mapping data structure between node and object."""
        raise NotImplementedError("Subclasses should implement this!")

    def persist(
        self, persist_dir: str = ..., obj_node_mapping_fname: str = ...
    ) -> None:
        """Persist objs."""
        raise NotImplementedError("Subclasses should implement this!")

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        obj_node_mapping_fname: str = DEFAULT_PERSIST_FNAME,
    ) -> "SQLTableNodeMapping":
        raise NotImplementedError(
            "This object node mapping does not support persist method."
        )
