from typing import Any, Callable, Optional, Sequence

from llama_index.core.tools import BaseTool
from llama_index.core.objects.base import SimpleObjectNodeMapping
from llama_index.core.objects.base_node_mapping import BaseObjectNodeMapping
from llama_index.core.objects.fn_node_mapping import FnNodeMapping
from llama_index.core.objects.tool_node_mapping import SimpleToolNodeMapping
from llama_index.core.schema import TextNode


def get_object_mapping(
    objects: Sequence[Any],
    from_node_fn: Optional[Callable[[TextNode], Any]] = None,
    to_node_fn: Optional[Callable[[Any], TextNode]] = None,
) -> BaseObjectNodeMapping:
    """Get object mapping according to object."""
    if from_node_fn is not None and to_node_fn is not None:
        return FnNodeMapping.from_objects(objects, from_node_fn, to_node_fn)
    elif all(isinstance(obj, BaseTool) for obj in objects):
        return SimpleToolNodeMapping.from_objects(objects)
    else:
        return SimpleObjectNodeMapping.from_objects(objects)
