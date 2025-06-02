"""Test node mapping."""

from llama_index.core import SQLDatabase
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.objects.base_node_mapping import SimpleObjectNodeMapping
from llama_index.core.objects.table_node_mapping import (
    SQLTableNodeMapping,
    SQLTableSchema,
)
from llama_index.core.objects.tool_node_mapping import SimpleToolNodeMapping
from llama_index.core.tools.function_tool import FunctionTool
from pytest_mock import MockerFixture


class _TestObject(BaseModel):
    """Test object for node mapping."""

    __test__ = False

    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return f"_TestObject(name='{self.name}')"


class _TestSQLDatabase(SQLDatabase):
    """Test object for SQL Table Schema Node Mapping."""

    def __init__(self) -> None:
        pass


def test_simple_object_node_mapping() -> None:
    """Test simple object node mapping."""
    strs = ["a", "b", "c"]
    node_mapping = SimpleObjectNodeMapping.from_objects(strs)
    assert node_mapping.to_node("a").text == "a"
    assert node_mapping.from_node(node_mapping.to_node("a")) == "a"

    objects = [_TestObject(name="a"), _TestObject(name="b"), _TestObject(name="c")]
    node_mapping = SimpleObjectNodeMapping.from_objects(objects)
    assert node_mapping.to_node(objects[0]).text == "_TestObject(name='a')"
    assert node_mapping.from_node(node_mapping.to_node(objects[0])) == objects[0]


def test_simple_object_node_mapping_persist() -> None:
    """Test persist/load."""
    strs = ["a", "b", "c"]
    node_mapping = SimpleObjectNodeMapping.from_objects(strs)
    node_mapping.persist()

    loaded_node_mapping = SimpleObjectNodeMapping.from_persist_dir()
    assert node_mapping.obj_node_mapping == loaded_node_mapping.obj_node_mapping


def test_tool_object_node_mapping() -> None:
    """Test tool object node mapping."""
    tool1 = FunctionTool.from_defaults(
        fn=lambda x: x,
        name="test_tool",
        description="test",
    )
    tool2 = FunctionTool.from_defaults(
        fn=lambda x, y: x + y, name="test_tool2", description="test"
    )

    node_mapping = SimpleToolNodeMapping.from_objects([tool1, tool2])
    # don't need to check for tool fn schema
    assert ("Tool name: test_tool\nTool description: test\n") in node_mapping.to_node(
        tool1
    ).get_text()
    assert node_mapping.from_node(node_mapping.to_node(tool1)) == tool1
    assert ("Tool name: test_tool2\nTool description: test\n") in node_mapping.to_node(
        tool2
    ).get_text()
    recon_tool2 = node_mapping.from_node(node_mapping.to_node(tool2))
    assert recon_tool2(1, 2).raw_output == 3

    tool3 = FunctionTool.from_defaults(
        fn=lambda x, y: x * y, name="test_tool3", description="test3"
    )
    node_mapping.add_object(tool3)
    assert ("Tool name: test_tool3\nTool description: test3\n") in node_mapping.to_node(
        tool3
    ).get_text()
    assert node_mapping.from_node(node_mapping.to_node(tool3)) == tool3


def test_sql_table_node_mapping_to_node(mocker: MockerFixture) -> None:
    """Test to add node for sql table node mapping object to ensure no 'None' values in metadata output to avoid issues with nulls when upserting to indexes."""
    mocker.patch(
        "llama_index.core.utilities.sql_wrapper.SQLDatabase.get_single_table_info",
        return_value="",
    )

    # Define two table schemas with one that does not have context str defined
    table1 = SQLTableSchema(table_name="table1")
    table2 = SQLTableSchema(table_name="table2", context_str="stuff here")
    tables = [table1, table2]

    # Create the mapping
    sql_database = _TestSQLDatabase()
    mapping = SQLTableNodeMapping(sql_database)

    # Create the nodes
    nodes = []
    for table in tables:
        node = mapping.to_node(table)
        nodes.append(node)

    # Make sure no None values are passed in otherwise PineconeVectorStore will fail the upsert
    for node in nodes:
        assert None not in node.metadata.values()
