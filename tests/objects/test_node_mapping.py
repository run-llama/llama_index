"""Test node mapping."""

from llama_index.objects.base_node_mapping import SimpleObjectNodeMapping
from llama_index.objects.tool_node_mapping import SimpleToolNodeMapping
from llama_index.tools.function_tool import FunctionTool

from pydantic import BaseModel


class TestObject(BaseModel):
    """Test object for node mapping."""

    name: str

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"TestObject(name='{self.name}')"


def test_simple_object_node_mapping():
    """Test simple object node mapping."""

    objects = ["a", "b", "c"]
    node_mapping = SimpleObjectNodeMapping.from_objects(objects)
    assert node_mapping.to_node("a").text == "a"
    assert node_mapping.from_node(node_mapping.to_node("a")) == "a"
    assert len(node_mapping._objs) == 3

    objects = [TestObject(name="a"), TestObject(name="b"), TestObject(name="c")]
    node_mapping = SimpleObjectNodeMapping.from_objects(objects)
    assert node_mapping.to_node(objects[0]).text == "TestObject(name='a')"
    assert node_mapping.from_node(node_mapping.to_node(objects[0])) == objects[0]
    assert len(node_mapping._objs) == 3


def test_tool_object_node_mapping():
    """Test tool object node mapping."""

    tool1 = FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")
    tool2 = FunctionTool.from_defaults(fn=lambda x, y: x + y, name="test_tool2")

    node_mapping = SimpleToolNodeMapping.from_objects([tool1, tool2])
    assert node_mapping.to_node(tool1).text == "test_tool"
    assert node_mapping.from_node(node_mapping.to_node(tool1)) == tool1
    assert node_mapping.to_node(tool2).text == "test_tool2"
    recon_tool2 = node_mapping.from_node(node_mapping.to_node(tool2))
    assert recon_tool2(1, 2) == 3

    tool3 = FunctionTool.from_defaults(fn=lambda x, y: x * y, name="test_tool3")
    node_mapping.add_object(tool3)
    assert node_mapping.to_node(tool3).text == "test_tool3"
    assert node_mapping.from_node(node_mapping.to_node(tool3)) == tool3
