"""Base object types."""

from typing import TypeVar, Generic, Sequence, Type
from abc import abstractmethod
from llama_index.data_structs.node import Node
from llama_index.indices.base import BaseIndex
from llama_index.indices.vector_store.base import VectorStoreIndex

OT = TypeVar("OT")


class BaseObjectNodeMapping(Generic[OT]):
    """Base object node mapping."""

    def __init__(self, objs: Sequence[OT]) -> None:
        self._objs = {hash(str(obj)): obj for obj in objs}

    def add_object(self, obj: OT) -> None:
        """Add an object."""
        self._objs[hash(str(obj))] = obj

    def to_nodes(self) -> Sequence[OT]:
        """Convert to nodes."""
        return self._objs

    @abstractmethod
    def to_node(self, obj: OT) -> OT:
        """To node."""
        pass

    @abstractmethod
    @classmethod
    def from_node(cls, node: Node) -> OT:
        """From node."""


class SimpleObjectNodeMapping(BaseObjectNodeMapping):
    """General node mapping that works for any obj."""

    def to_node(self, obj) -> Node:
        return Node(text=str(obj))

    def from_node(self, node: Node) -> OT:
        return self._objs[hash(node.text)]
