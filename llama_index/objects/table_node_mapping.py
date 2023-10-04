"""Table node mapping."""

from typing import Any, Optional, Sequence

from llama_index.bridge.pydantic import BaseModel
from llama_index.objects.base_node_mapping import BaseObjectNodeMapping
from llama_index.schema import BaseNode, TextNode
from llama_index.utilities.sql_wrapper import SQLDatabase


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
        if obj.context_str is not None:
            table_text += f"Context of table {obj.table_name}:\n"
            table_text += obj.context_str

        return TextNode(
            text=table_text,
            metadata={"name": obj.table_name, "context": obj.context_str},
            excluded_embed_metadata_keys=["name", "context"],
            excluded_llm_metadata_keys=["name", "context"],
        )

    def _from_node(self, node: BaseNode) -> SQLTableSchema:
        """From node."""
        if node.metadata is None:
            raise ValueError("Metadata must be set")
        return SQLTableSchema(
            table_name=node.metadata["name"], context_str=node.metadata["context"]
        )
