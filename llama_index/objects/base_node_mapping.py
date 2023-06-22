"""Base object types."""

from typing import TypeVar, Generic, Sequence, Any
from abc import abstractmethod
from llama_index.data_structs.node import Node

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

    def validate_object(self, obj: OT) -> None:
        """Validate object."""
        pass

    def add_object(self, obj: OT) -> None:
        """Add object.

        Only needs to be specified if the node mapping
        needs to be initialized with a list of objects.

        """
        self.validate_object(obj)
        self._add_object(obj)

    @abstractmethod
    def _add_object(self, obj: OT) -> None:
        """Add object.

        Only needs to be specified if the node mapping
        needs to be initialized with a list of objects.

        """

    @abstractmethod
    def to_node(self, obj: OT) -> Node:
        """To node."""
        pass

    def to_nodes(self, objs: Sequence[OT]) -> Sequence[Node]:
        return [self.to_node(obj) for obj in objs]

    def from_node(self, node: Node) -> OT:
        """From node."""
        obj = self._from_node(node)
        self.validate_object(obj)
        return obj

    @abstractmethod
    def _from_node(self, node: Node) -> OT:
        """From node."""


class SimpleObjectNodeMapping(BaseObjectNodeMapping[Any]):
    """General node mapping that works for any obj.

    More specifically, any object with a meaningful string representation.

    """

    def __init__(self, objs: Sequence[Any]) -> None:
        for obj in objs:
            self.validate_object(obj)
        self._objs = {hash(str(obj)): obj for obj in objs}

    @classmethod
    def from_objects(
        cls, objs: Sequence[Any], *args: Any, **kwargs: Any
    ) -> "BaseObjectNodeMapping":
        return cls(objs)

    def _add_object(self, obj: Any) -> None:
        self._objs[hash(str(obj))] = obj

    def to_node(self, obj: Any) -> Node:
        return Node(text=str(obj))

    def _from_node(self, node: Node) -> Any:
        return self._objs[hash(node.text)]
