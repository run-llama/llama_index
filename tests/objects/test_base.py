"""Test object index."""

from llama_index.indices.list.base import SummaryIndex
from llama_index.objects.base import ObjectIndex
from llama_index.objects.base_node_mapping import SimpleObjectNodeMapping
from llama_index.objects.tool_node_mapping import SimpleToolNodeMapping
from llama_index.service_context import ServiceContext
from llama_index.tools.function_tool import FunctionTool


def test_object_index(mock_service_context: ServiceContext) -> None:
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


def test_object_index_persist(mock_service_context: ServiceContext) -> None:
    """Test object index persist/load."""
    object_mapping = SimpleObjectNodeMapping.from_objects(["a", "b", "c"])
    obj_index = ObjectIndex.from_objects(
        ["a", "b", "c"], object_mapping, index_cls=SummaryIndex
    )
    obj_index.persist()

    reloaded_obj_index = ObjectIndex.from_persist_dir(
        object_node_mapping_type=SimpleObjectNodeMapping
    )
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


def test_object_index_with_tools(mock_service_context: ServiceContext) -> None:
    """Test object index with tools."""
    tool1 = FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")
    tool2 = FunctionTool.from_defaults(fn=lambda x, y: x + y, name="test_tool2")

    object_mapping = SimpleToolNodeMapping.from_objects([tool1, tool2])

    obj_retriever = ObjectIndex.from_objects(
        [tool1, tool2], object_mapping, index_cls=SummaryIndex
    )
    assert obj_retriever.as_retriever().retrieve("test") == [tool1, tool2]
