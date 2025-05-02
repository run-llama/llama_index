"""Table node mapping."""

from typing import Any, Callable, Dict, Sequence

from llama_index.core.objects.base_node_mapping import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    BaseObjectNodeMapping,
)
from llama_index.core.schema import BaseNode


class FnNodeMapping(BaseObjectNodeMapping[Any]):
    """Fn node mapping."""

    def __init__(
        self,
        from_node_fn: Callable[[BaseNode], Any],
        to_node_fn: Callable[[Any], BaseNode],
    ) -> None:
        self._to_node_fn = to_node_fn
        self._from_node_fn = from_node_fn

    @classmethod
    def from_objects(  # type: ignore
        cls,
        objs: Sequence[Any],
        from_node_fn: Callable[[BaseNode], Any],
        to_node_fn: Callable[[Any], BaseNode],
        *args: Any,
        **kwargs: Any,
    ) -> "BaseObjectNodeMapping":
        """Initialize node mapping."""
        return cls(from_node_fn, to_node_fn)

    def _add_object(self, obj: Any) -> None:
        """Add object. NOTE: unused."""

    def to_node(self, obj: Any) -> BaseNode:
        """To node."""
        return self._to_node_fn(obj)

    def _from_node(self, node: BaseNode) -> Any:
        """From node."""
        return self._from_node_fn(node)

    @property
    def obj_node_mapping(self) -> Dict[int, Any]:
        """The mapping data structure between node and object."""
        raise NotImplementedError("FnNodeMapping does not support obj_node_mapping")

    def persist(
        self, persist_dir: str = ..., obj_node_mapping_fname: str = ...
    ) -> None:
        """Persist objs."""
        raise NotImplementedError("FnNodeMapping does not support persist method.")

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        obj_node_mapping_fname: str = DEFAULT_PERSIST_FNAME,
    ) -> "FnNodeMapping":
        raise NotImplementedError("FnNodeMapping does not support persist method.")
