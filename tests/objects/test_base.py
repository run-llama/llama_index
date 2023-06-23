"""Test object index."""

from llama_index.objects.base import ObjectIndex
from llama_index.indices.list.base import ListIndex

from llama_index.objects.base_node_mapping import SimpleObjectNodeMapping
from llama_index.objects.tool_node_mapping import SimpleToolNodeMapping
from llama_index.tools.function_tool import FunctionTool
from llama_index.indices.service_context import ServiceContext


def test_object_index(mock_service_context: ServiceContext) -> None:
    """Test object index."""

    object_mapping = SimpleObjectNodeMapping.from_objects(["a", "b", "c"])
    obj_index = ObjectIndex.from_objects(
        ["a", "b", "c"], object_mapping, index_cls=ListIndex
    )
    # should just retrieve everything
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c"]

    # test adding an object
    obj_index.insert_object("d")
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c", "d"]


def test_object_index_with_tools(mock_service_context: ServiceContext) -> None:
    """Test object index with tools."""

    tool1 = FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")
    tool2 = FunctionTool.from_defaults(fn=lambda x, y: x + y, name="test_tool2")

    object_mapping = SimpleToolNodeMapping.from_objects([tool1, tool2])

    obj_retriever = ObjectIndex.from_objects(
        [tool1, tool2], object_mapping, index_cls=ListIndex
    )
    assert obj_retriever.as_retriever().retrieve("test") == [tool1, tool2]
