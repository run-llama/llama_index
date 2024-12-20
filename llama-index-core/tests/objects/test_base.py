"""Test object index."""

from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.objects.base import ObjectIndex
from llama_index.core.objects.base_node_mapping import SimpleObjectNodeMapping
from llama_index.core.objects.tool_node_mapping import SimpleToolNodeMapping
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.schema import TextNode


def test_object_index() -> None:
    """Test object index."""
    object_mapping = SimpleObjectNodeMapping.from_objects(["a", "b", "c"])
    obj_index = ObjectIndex.from_objects(
        ["a", "b", "c"], object_mapping, index_cls=SummaryIndex
    )
    # should just retrieve everything
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c"]

    # test adding an object
    obj_index.insert_object("d")
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c", "d"]


def test_object_index_default_mapping() -> None:
    """Test object index."""
    obj_index = ObjectIndex.from_objects(["a", "b", "c"], index_cls=SummaryIndex)
    # should just retrieve everything
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c"]

    # test adding an object
    obj_index.insert_object("d")
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c", "d"]


def test_object_index_fn_mapping() -> None:
    """Test object index."""
    objects = {obj: obj for obj in ["a", "b", "c", "d"]}
    print(objects)

    def to_node_fn(obj: str) -> TextNode:
        return TextNode(id_=obj, text=obj)

    def from_node_fn(node: TextNode) -> str:
        return objects[node.id_]

    obj_index = ObjectIndex.from_objects(
        ["a", "b", "c"],
        index_cls=SummaryIndex,
        from_node_fn=from_node_fn,
        to_node_fn=to_node_fn,
    )

    # should just retrieve everything
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c"]

    # test adding an object
    obj_index.insert_object("d")
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c", "d"]


def test_object_index_persist() -> None:
    """Test object index persist/load."""
    object_mapping = SimpleObjectNodeMapping.from_objects(["a", "b", "c"])
    obj_index = ObjectIndex.from_objects(
        ["a", "b", "c"], object_mapping, index_cls=SummaryIndex
    )
    obj_index.persist()

    reloaded_obj_index = ObjectIndex.from_persist_dir()
    assert obj_index._index.index_id == reloaded_obj_index._index.index_id
    assert obj_index._index.index_struct == reloaded_obj_index._index.index_struct
    assert (
        obj_index._object_node_mapping.obj_node_mapping
        == reloaded_obj_index._object_node_mapping.obj_node_mapping
    )

    # version where user passes in the object_node_mapping
    reloaded_obj_index = ObjectIndex.from_persist_dir(
        object_node_mapping=object_mapping
    )
    assert obj_index._index.index_id == reloaded_obj_index._index.index_id
    assert obj_index._index.index_struct == reloaded_obj_index._index.index_struct
    assert (
        obj_index._object_node_mapping.obj_node_mapping
        == reloaded_obj_index._object_node_mapping.obj_node_mapping
    )


def test_object_index_with_tools() -> None:
    """Test object index with tools."""
    tool1 = FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")
    tool2 = FunctionTool.from_defaults(fn=lambda x, y: x + y, name="test_tool2")

    object_mapping = SimpleToolNodeMapping.from_objects([tool1, tool2])

    obj_retriever = ObjectIndex.from_objects(
        [tool1, tool2], object_mapping, index_cls=SummaryIndex
    )
    assert obj_retriever.as_retriever().retrieve("test") == [tool1, tool2]
