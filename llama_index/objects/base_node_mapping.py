"""Base object types."""

from abc import abstractmethod
from typing import Any, Generic, Optional, Sequence, TypeVar

from llama_index.schema import BaseNode, MetadataMode, TextNode

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

    def validate_object(self, obj: OT) -> None:
        """Validate object."""

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
    def to_node(self, obj: OT) -> TextNode:
        """To node."""

    def to_nodes(self, objs: Sequence[OT]) -> Sequence[TextNode]:
        return [self.to_node(obj) for obj in objs]

    def from_node(self, node: BaseNode) -> OT:
        """From node."""
        obj = self._from_node(node)
        self.validate_object(obj)
        return obj

    @abstractmethod
    def _from_node(self, node: BaseNode) -> OT:
        """From node."""


class SimpleObjectNodeMapping(BaseObjectNodeMapping[Any]):
    """General node mapping that works for any obj.

    More specifically, any object with a meaningful string representation.

    """

    def __init__(self, objs: Optional[Sequence[Any]] = None) -> None:
        objs = objs or []
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

    def to_node(self, obj: Any) -> TextNode:
        return TextNode(text=str(obj))

    def _from_node(self, node: BaseNode) -> Any:
        return self._objs[hash(node.get_content(metadata_mode=MetadataMode.NONE))]
