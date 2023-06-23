"""Table node mapping."""

from llama_index.objects.base_node_mapping import BaseObjectNodeMapping

from typing import List, Any, Sequence, Optional
from pydantic import BaseModel
from sqlalchemy import Table
from llama_index.langchain_helpers.sql_wrapper import SQLDatabase
from llama_index.data_structs.node import Node


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
        sql_database: SQLDatabase,
        *args: Any,
        **kwargs: Any,
    ) -> "BaseObjectNodeMapping":
        """Initialize node mapping."""
        # ignore objs, since we are building from sql_database
        return cls(sql_database)

    def _add_object(self, obj: SQLTableSchema) -> None:
        raise NotImplementedError

    def to_node(self, obj: SQLTableSchema) -> Node:
        """To node."""
        # taken from existing schema logic
        table_text = (
            f"Schema of table {obj.table_name}:\n"
            f"{self._sql_database.get_single_table_info(obj.table_name)}\n"
        )
        if obj.context_str is not None:
            table_text += f"Context of table {obj.table_name}:\n"
            table_text += obj.context_str

        return Node(
            text=table_text,
            node_info={"name": obj.table_name, "context": obj.context_str},
        )

    def _from_node(self, node: Node) -> SQLTableSchema:
        """From node."""
        if node.node_info is None:
            raise ValueError("Node info must be set")
        return SQLTableSchema(
            table_name=node.node_info["name"], context_str=node.node_info["context"]
        )
