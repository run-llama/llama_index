"""Base object types."""

from typing import TypeVar, Generic, Sequence, Type, Any
from abc import abstractmethod
from llama_index.data_structs.node import Node
from llama_index.indices.base import BaseIndex
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.tools.types import BaseTool

OT = TypeVar("OT")


class BaseObjectNodeMapping(Generic[OT]):
    """Base object node mapping."""

    @classmethod
    @abstractmethod
    def from_objects(
        cls, objs: Sequence[OT], *args: Any, **kwargs: Any
    ) -> "BaseObjectNodeMapping":
        """Initialize node mapping from a list of objects.

        Only needs to be specified if the node mapping
        needs to be initialized with a list of objects.

        """
        pass

    @abstractmethod
    def add_object(self, obj: OT) -> None:
        """Add object.

        Only needs to be specified if the node mapping
        needs to be initialized with a list of objects.

        """
        pass

    @abstractmethod
    def to_node(self, obj: OT) -> OT:
        """To node."""
        pass

    @abstractmethod
    def from_node(self, node: Node) -> OT:
        """From node."""


class SimpleObjectNodeMapping(BaseObjectNodeMapping):
    """General node mapping that works for any obj."""

    def __init__(self, objs: Sequence[OT]) -> None:
        self._objs = {hash(obj): obj for obj in objs}

    @classmethod
    def from_objects(
        cls, objs: Sequence[OT], *args: Any, **kwargs: Any
    ) -> "BaseObjectNodeMapping":
        return cls(objs)

    def add_object(self, obj: OT) -> None:
        self._objs[hash(obj)] = obj

    def to_node(self, obj: OT) -> Node:
        return Node(text=str(obj))

    def from_node(self, node: Node) -> OT:
        return self._objs[hash(node.text)]
