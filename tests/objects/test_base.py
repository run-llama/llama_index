"""Test object index."""

from typing import List, Dict, Any
from llama_index.data_structs.node import NodeWithScore, Node
from llama_index.indices.query.schema import QueryBundle
from llama_index.objects.base import ObjectIndex, ObjectRetriever
from llama_index.indices.list.base import ListIndex

from llama_index.objects.base_node_mapping import SimpleObjectNodeMapping
from llama_index.objects.tool_node_mapping import SimpleToolNodeMapping
from llama_index.tools.function_tool import FunctionTool

from llama_index.indices.base_retriever import BaseRetriever


def test_object_index():
    """Test object index."""

    obj_index = ObjectIndex.from_objects(
        ["a", "b", "c"], index_cls=ListIndex, object_mapping_cls=SimpleObjectNodeMapping
    )
    # should just retrieve everything
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c"]

    # test adding an object
    obj_index.insert_object("d")
    assert obj_index.as_retriever().retrieve("test") == ["a", "b", "c", "d"]


def test_object_index_with_tools():
    """Test object index with tools."""

    tool1 = FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")
    tool2 = FunctionTool.from_defaults(fn=lambda x, y: x + y, name="test_tool2")

    obj_retriever = ObjectIndex.from_objects(
        [tool1, tool2], index_cls=ListIndex, object_mapping_cls=SimpleToolNodeMapping
    )
    assert obj_retriever.as_retriever().retrieve("test") == [tool1, tool2]
