"""Test tools."""

import json
from typing import List, Optional

import pytest
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import TextBlock, ImageBlock
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.schema import Document, TextNode
from llama_index.core.workflow.context import Context
from llama_index.core.workflow import Context

try:
    import langchain  # pants: no-infer-dep
except ImportError:
    langchain = None  # type: ignore


def tmp_function(x: int) -> str:
    return str(x)


async def async_tmp_function(x: int) -> str:
    return "async_" + str(x)


def test_function_tool() -> None:
    """Test function tool."""
    function_tool = FunctionTool.from_defaults(
        lambda x: str(x), name="foo", description="bar"
    )
    assert function_tool.metadata.name == "foo"
    assert function_tool.metadata.description == "bar"
    assert function_tool.metadata.fn_schema is not None
    actual_schema = function_tool.metadata.fn_schema.model_json_schema()
    # note: no type
    assert "x" in actual_schema["properties"]

    result = function_tool(1)
    assert str(result) == "1"

    # test adding typing to function

    function_tool = FunctionTool.from_defaults(
        tmp_function, name="foo", description="bar"
    )
    assert function_tool.metadata.fn_schema is not None
    actual_schema = function_tool.metadata.fn_schema.model_json_schema()
    assert actual_schema["properties"]["x"]["type"] == "integer"

    # should not have ctx param requirements
    assert function_tool.ctx_param_name is None
    assert not function_tool.requires_context


@pytest.mark.skipif(langchain is None, reason="langchain not installed")
def test_function_tool_to_langchain() -> None:
    function_tool = FunctionTool.from_defaults(
        tmp_function, name="foo", description="bar"
    )

    # test to langchain
    # NOTE: can't take in a function with int args
    langchain_tool = function_tool.to_langchain_tool()
    result = langchain_tool.run("1")
    assert result == "1"

    # test langchain structured tool
    class TestSchema(BaseModel):
        x: int
        y: int

    function_tool = FunctionTool.from_defaults(
        lambda x, y: str(x) + "," + str(y),
        name="foo",
        description="bar",
        fn_schema=TestSchema,
    )
    assert str(function_tool(1, 2)) == "1,2"
    langchain_tool2 = function_tool.to_langchain_structured_tool()
    assert langchain_tool2.run({"x": 1, "y": 2}) == "1,2"
    assert langchain_tool2.args_schema == TestSchema


@pytest.mark.asyncio
async def test_function_tool_async() -> None:
    """Test function tool async."""
    function_tool = FunctionTool.from_defaults(
        fn=tmp_function, async_fn=async_tmp_function, name="foo", description="bar"
    )
    assert function_tool.metadata.fn_schema is not None
    actual_schema = function_tool.metadata.fn_schema.model_json_schema()
    assert actual_schema["properties"]["x"]["type"] == "integer"

    assert str(function_tool(2)) == "2"
    assert str(await function_tool.acall(2)) == "async_2"


@pytest.mark.skipif(langchain is None, reason="langchain not installed")
@pytest.mark.asyncio
async def test_function_tool_async_langchain() -> None:
    function_tool = FunctionTool.from_defaults(
        fn=tmp_function, async_fn=async_tmp_function, name="foo", description="bar"
    )

    # test to langchain
    # NOTE: can't take in a function with int args
    langchain_tool = function_tool.to_langchain_tool()
    result = await langchain_tool.arun("1")
    assert result == "async_1"

    # test langchain structured tool
    class TestSchema(BaseModel):
        x: int
        y: int

    def structured_tmp_function(x: int, y: int) -> str:
        return str(x) + "," + str(y)

    async def async_structured_tmp_function(x: int, y: int) -> str:
        return "async_" + str(x) + "," + str(y)

    function_tool = FunctionTool.from_defaults(
        fn=structured_tmp_function,
        async_fn=async_structured_tmp_function,
        name="foo",
        description="bar",
        fn_schema=TestSchema,
    )
    assert str(await function_tool.acall(1, 2)) == "async_1,2"
    langchain_tool2 = function_tool.to_langchain_structured_tool()
    assert (await langchain_tool2.arun({"x": 1, "y": 2})) == "async_1,2"
    assert langchain_tool2.args_schema == TestSchema


@pytest.mark.asyncio
async def test_function_tool_async_defaults() -> None:
    """Test async calls to function tool when only sync function is given."""
    function_tool = FunctionTool.from_defaults(
        fn=tmp_function, name="foo", description="bar"
    )
    assert function_tool.metadata.fn_schema is not None
    actual_schema = function_tool.metadata.fn_schema.model_json_schema()
    assert actual_schema["properties"]["x"]["type"] == "integer"


@pytest.mark.skipif(langchain is None, reason="langchain not installed")
@pytest.mark.asyncio
async def test_function_tool_async_defaults_langchain() -> None:
    function_tool = FunctionTool.from_defaults(
        fn=tmp_function, name="foo", description="bar"
    )

    # test to langchain
    # NOTE: can't take in a function with int args
    langchain_tool = function_tool.to_langchain_tool()
    result = await langchain_tool.arun("1")
    assert result == "1"


from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.tools import RetrieverTool, ToolMetadata


def test_retreiver_tool() -> None:
    doc1 = Document(
        text=("# title1:Hello world.\nThis is a test.\n"),
        metadata={"file_path": "/data/personal/essay.md"},
    )

    doc2 = Document(
        text=("# title2:This is another test.\nThis is a test v2."),
        metadata={"file_path": "/data/personal/essay.md"},
    )
    vs_index = VectorStoreIndex.from_documents([doc1, doc2])
    vs_retriever = vs_index.as_retriever()
    vs_ret_tool = RetrieverTool(
        retriever=vs_retriever,
        metadata=ToolMetadata(
            name="knowledgebase",
            description="test",
        ),
    )
    output = vs_ret_tool.call("arg1", "arg2", key1="v1", key2="v2")
    formated_doc = (
        "file_path: /data/personal/essay.md\n\n# title1:Hello world.\nThis is a test."
    )
    assert formated_doc in output.content


def test_tool_fn_schema() -> None:
    class TestSchema(BaseModel):
        input: Optional[str]
        page_list: List[int]

    metadata = ToolMetadata(
        name="a useful tool", description="test", fn_schema=TestSchema
    )
    parameter_dict = json.loads(metadata.fn_schema_str)
    assert set(parameter_dict.keys()) == {"type", "properties", "required"}


def test_function_tool_partial_params_schema() -> None:
    def test_function(x: int, y: int) -> str:
        return f"x: {x}, y: {y}"

    tool = FunctionTool.from_defaults(test_function, partial_params={"y": 2})
    assert tool.metadata.fn_schema is not None
    actual_schema = tool.metadata.fn_schema.model_json_schema()
    assert actual_schema["properties"]["x"]["type"] == "integer"
    assert "y" not in actual_schema["properties"]


def test_function_tool_partial_params() -> None:
    def test_function(x: int, y: int) -> str:
        return f"x: {x}, y: {y}"

    tool = FunctionTool.from_defaults(test_function, partial_params={"y": 2})
    assert tool(x=1).raw_output == "x: 1, y: 2"
    assert tool(x=1).raw_input == {"args": (), "kwargs": {"x": 1, "y": 2}}
    assert tool(x=1, y=3).raw_output == "x: 1, y: 3"
    assert tool(x=1, y=3).raw_input == {"args": (), "kwargs": {"x": 1, "y": 3}}


@pytest.mark.asyncio
async def test_function_tool_partial_params_async() -> None:
    async def test_function(x: int, y: int) -> str:
        return f"x: {x}, y: {y}"

    tool = FunctionTool.from_defaults(test_function, partial_params={"y": 2})
    assert (await tool.acall(x=1)).raw_output == "x: 1, y: 2"
    assert (await tool.acall(x=1)).raw_input == {"args": (), "kwargs": {"x": 1, "y": 2}}
    assert (await tool.acall(x=1, y=3)).raw_output == "x: 1, y: 3"
    assert (await tool.acall(x=1, y=3)).raw_input == {
        "args": (),
        "kwargs": {"x": 1, "y": 3},
    }


def test_function_tool_ctx_param() -> None:
    def test_function(x: int, ctx: Context) -> str:
        return f"x: {x}, ctx: {ctx}"

    tool = FunctionTool.from_defaults(test_function)
    assert tool.metadata.fn_schema is not None
    assert tool.ctx_param_name == "ctx"
    assert tool.requires_context

    actual_schema = tool.metadata.fn_schema.model_json_schema()
    assert "ctx" not in actual_schema["properties"]
    assert len(actual_schema["properties"]) == 1
    assert actual_schema["properties"]["x"]["type"] == "integer"


def test_function_tool_ctx_generic_param() -> None:
    class MyState(BaseModel):
        name: str = Field(default="Logan")

    async def test_function(x: int, ctx_arg: Context[MyState]) -> str:
        return f"x: {x}, ctx: {ctx_arg}"

    tool = FunctionTool.from_defaults(test_function)
    assert tool.metadata.fn_schema is not None
    assert tool.ctx_param_name == "ctx_arg"
    assert tool.requires_context

    actual_schema = tool.metadata.fn_schema.model_json_schema()
    assert "ctx_arg" not in actual_schema["properties"]
    assert len(actual_schema["properties"]) == 1
    assert actual_schema["properties"]["x"]["type"] == "integer"


def test_function_tool_self_param() -> None:
    class FunctionHolder:
        def test_function(self, x: int, ctx: Context) -> str:
            return f"x: {x}, ctx: {ctx}"

    tool = FunctionTool.from_defaults(FunctionHolder.test_function)
    assert tool.metadata.fn_schema is not None
    assert tool.ctx_param_name == "ctx"
    assert tool.requires_context

    actual_schema = tool.metadata.fn_schema.model_json_schema()
    assert "self" not in actual_schema["properties"]
    assert "ctx" not in actual_schema["properties"]
    assert "x" in actual_schema["properties"]


@pytest.mark.asyncio
async def test_function_tool_output_blocks() -> None:
    def test_function() -> str:
        return [
            TextBlock(text="Hello"),
            ImageBlock(url="https://example.com/image.png"),
        ]

    tool = FunctionTool.from_defaults(test_function)

    tool_output = tool.call()

    assert len(tool_output.blocks) == 2
    assert tool_output.content == "Hello"

    tool_output = await tool.acall()

    assert len(tool_output.blocks) == 2
    assert tool_output.content == "Hello"


@pytest.mark.asyncio
async def test_function_tool_output_single_block() -> None:
    def test_function() -> str:
        return TextBlock(text="Hello")

    tool = FunctionTool.from_defaults(test_function)

    tool_output = tool.call()

    assert len(tool_output.blocks) == 1
    assert tool_output.content == "Hello"

    tool_output = await tool.acall()

    assert len(tool_output.blocks) == 1
    assert tool_output.content == "Hello"


def test_fn_schema_docstring_descriptions():
    def sample_tool(a: int, b: Optional[str] = None) -> str:
        """
        A test tool function.

        :param a: an integer value
        :param b: an optional string
        :return: a string output
        """
        return f"{a}-{b}"

    metadata = ToolMetadata(name="sample_tool", description="A test tool.")
    tool = FunctionTool.from_defaults(fn=sample_tool, tool_metadata=None)

    fn_schema = tool.metadata.fn_schema
    assert fn_schema is not None, "Expected fn_schema to be generated"

    fields = fn_schema.model_fields
    assert "a" in fields
    assert "b" in fields

    assert fields["a"].description == "an integer value"
    assert fields["b"].description == "an optional string"


def test_docstring_param_extraction_javadoc_style():
    def tool_fn(foo: int, bar: str) -> str:
        """
        Test tool.

        @param foo value for foo
        @param bar value for bar
        @return result string
        """
        return f"{foo}-{bar}"

    tool = FunctionTool.from_defaults(fn=tool_fn)
    fields = tool.metadata.fn_schema.model_fields

    assert fields["foo"].description == "value for foo"
    assert fields["bar"].description == "value for bar"


def test_docstring_param_extraction_google_style():
    def tool_fn(a: int, b: str) -> str:
        """
        Test tool.

        Args:
            a (int): integer input
            b (str): string input

        Returns:
            str: output string

        """
        return f"{a}-{b}"

    tool = FunctionTool.from_defaults(fn=tool_fn)
    fields = tool.metadata.fn_schema.model_fields

    assert fields["a"].description == "integer input"
    assert fields["b"].description == "string input"


def test_docstring_ignores_unknown_params():
    def tool_fn(a: int) -> str:
        """
        Test tool.

        :param a: valid param
        :param unknown: should be ignored
        """
        return str(a)

    tool = FunctionTool.from_defaults(fn=tool_fn)
    fields = tool.metadata.fn_schema.model_fields

    assert "a" in fields
    assert fields["a"].description == "valid param"
    assert "unknown" not in fields


def test_docstring_with_self_and_context():
    class MyTool:
        def my_method(self, ctx: Context, a: int) -> str:
            """
            Tool with self and context.

            :param a: some input value
            """
            return str(a)

    tool = FunctionTool.from_defaults(fn=MyTool().my_method)
    fields = tool.metadata.fn_schema.model_fields

    assert "a" in fields
    assert fields["a"].description == "some input value"
    assert "self" not in fields


def test_function_tool_output_document_and_nodes():
    def get_document() -> Document:
        return Document(text="Hello" * 1024)

    def get_node() -> TextNode:
        return TextNode(text="Hello" * 1024)

    def get_documents() -> List[Document]:
        return [Document(text="Hello" * 1024), Document(text="World" * 1024)]

    def get_nodes() -> List[TextNode]:
        return [TextNode(text="Hello" * 1024), TextNode(text="World" * 1024)]

    tool = FunctionTool.from_defaults(get_document)
    assert tool.call().content == "Hello" * 1024

    tool = FunctionTool.from_defaults(get_node)
    assert tool.call().content == "Hello" * 1024

    tool = FunctionTool.from_defaults(get_documents)
    assert "Hello" * 1024 in tool.call().content
    assert "World" * 1024 in tool.call().content

    tool = FunctionTool.from_defaults(get_nodes)
    assert "Hello" * 1024 in tool.call().content
    assert "World" * 1024 in tool.call().content
