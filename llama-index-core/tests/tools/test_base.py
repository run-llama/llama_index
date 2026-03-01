"""Test tools."""

import json
from typing import List, Optional

import pytest
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import TextBlock, ImageBlock
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import ToolMiddleware
from llama_index.core.tools.middleware import (
    ParameterInjectionMiddleware,
    OutputFilterMiddleware,
)
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


# --- Middleware Tests ---


class AppendMiddleware(ToolMiddleware):
    """Test middleware that appends a suffix to a 'text' kwarg."""

    def __init__(self, suffix: str) -> None:
        self._suffix = suffix

    def process_input(self, tool, kwargs):
        if "text" in kwargs:
            kwargs = {**kwargs, "text": kwargs["text"] + self._suffix}
        return kwargs

    def process_output(self, tool, output):
        if isinstance(output, str):
            return output + self._suffix
        return output


def test_middleware_single() -> None:
    """Test single middleware applied to a tool."""

    def greet(text: str) -> str:
        return f"Hello, {text}"

    mw = AppendMiddleware("!")
    tool = FunctionTool.from_defaults(greet, middlewares=[mw])

    result = tool(text="world")
    # Input: "world" -> "world!" via input middleware
    # Output: "Hello, world!" -> "Hello, world!!" via output middleware
    assert result.raw_output == "Hello, world!!"


def test_middleware_chain_order() -> None:
    """Test that multiple middleware are applied in correct order."""

    def echo(text: str) -> str:
        return text

    mw1 = AppendMiddleware("-A")
    mw2 = AppendMiddleware("-B")
    tool = FunctionTool.from_defaults(echo, middlewares=[mw1, mw2])

    result = tool(text="start")
    # Input: "start" -> "start-A" (mw1) -> "start-A-B" (mw2)
    # Output: "start-A-B" -> "start-A-B-B" (mw2 reverse) -> "start-A-B-B-A" (mw1 reverse)
    assert result.raw_output == "start-A-B-B-A"


@pytest.mark.asyncio
async def test_middleware_async() -> None:
    """Test middleware works with async tool calls."""

    async def greet(text: str) -> str:
        return f"Hello, {text}"

    mw = AppendMiddleware("!")
    tool = FunctionTool.from_defaults(greet, middlewares=[mw])

    result = await tool.acall(text="world")
    assert result.raw_output == "Hello, world!!"


def test_middleware_backward_compatibility() -> None:
    """Test that tools without middleware still work exactly as before."""

    def add(x: int, y: int) -> str:
        return f"{x + y}"

    tool = FunctionTool.from_defaults(add)
    result = tool(x=1, y=2)
    assert result.raw_output == "3"


def test_parameter_injection_middleware_enforce() -> None:
    """Test that ParameterInjectionMiddleware enforces parameters."""

    def query(text: str, api_key: str) -> str:
        return f"text={text}, key={api_key}"

    mw = ParameterInjectionMiddleware(
        params={"api_key": "trusted-key"},
        enforce={"api_key"},
    )
    tool = FunctionTool.from_defaults(query, middlewares=[mw])

    # LLM tries to override api_key — middleware should enforce it
    result = tool(text="hello", api_key="hacked-key")
    assert result.raw_output == "text=hello, key=trusted-key"


def test_parameter_injection_middleware_default_only() -> None:
    """Test that non-enforced params act as defaults."""

    def query(text: str, limit: int = 10) -> str:
        return f"text={text}, limit={limit}"

    mw = ParameterInjectionMiddleware(
        params={"limit": 5},
        enforce=set(),  # Nothing enforced — all are defaults
    )
    tool = FunctionTool.from_defaults(query, middlewares=[mw])

    # Without LLM override: uses middleware default
    assert tool(text="hello").raw_output == "text=hello, limit=5"

    # With LLM override: LLM value wins
    assert tool(text="hello", limit=20).raw_output == "text=hello, limit=20"


def test_parameter_injection_middleware_enforce_all_by_default() -> None:
    """Test that all params are enforced when enforce=None."""

    def query(x: int, y: int) -> str:
        return f"x={x}, y={y}"

    mw = ParameterInjectionMiddleware(params={"y": 99})
    tool = FunctionTool.from_defaults(query, middlewares=[mw])

    # enforce=None means all params enforced — LLM can't override y
    assert tool(x=1, y=0).raw_output == "x=1, y=99"


def test_output_filter_middleware_allowed_fields() -> None:
    """Test output filtering with allowed_fields."""

    def get_user() -> dict:
        return {
            "id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "internal_score": 42.5,
            "password_hash": "abc123",
        }

    mw = OutputFilterMiddleware(allowed_fields={"id", "name"})
    tool = FunctionTool.from_defaults(get_user, middlewares=[mw])

    result = tool()
    assert result.raw_output == {"id": 1, "name": "Alice"}


def test_output_filter_middleware_excluded_fields() -> None:
    """Test output filtering with excluded_fields."""

    def get_user() -> dict:
        return {"id": 1, "name": "Alice", "password_hash": "abc123"}

    mw = OutputFilterMiddleware(excluded_fields={"password_hash"})
    tool = FunctionTool.from_defaults(get_user, middlewares=[mw])

    result = tool()
    assert result.raw_output == {"id": 1, "name": "Alice"}


def test_output_filter_middleware_list_of_dicts() -> None:
    """Test output filtering on a list of dicts."""

    def get_users() -> list:
        return [
            {"id": 1, "name": "Alice", "secret": "s1"},
            {"id": 2, "name": "Bob", "secret": "s2"},
        ]

    mw = OutputFilterMiddleware(allowed_fields={"id", "name"})
    tool = FunctionTool.from_defaults(get_users, middlewares=[mw])

    result = tool()
    assert result.raw_output == [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]


def test_output_filter_middleware_validation() -> None:
    """Test that both allowed_fields and excluded_fields raises ValueError."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        OutputFilterMiddleware(allowed_fields={"a"}, excluded_fields={"b"})


def test_middleware_with_partial_params() -> None:
    """Test that middleware works together with partial_params."""

    def query(text: str, user_id: str, limit: int = 10) -> str:
        return f"text={text}, user={user_id}, limit={limit}"

    # partial_params provides user_id, middleware enforces limit
    mw = ParameterInjectionMiddleware(params={"limit": 3}, enforce={"limit"})
    tool = FunctionTool.from_defaults(
        query,
        partial_params={"user_id": "u1"},
        middlewares=[mw],
    )

    result = tool(text="hello", limit=100)
    # limit=100 from LLM -> middleware enforces limit=3
    # partial_params merges user_id=u1
    assert result.raw_output == "text=hello, user=u1, limit=3"


@pytest.mark.asyncio
async def test_parameter_injection_middleware_async() -> None:
    """Test ParameterInjectionMiddleware works with async calls."""

    async def query(text: str, api_key: str) -> str:
        return f"text={text}, key={api_key}"

    mw = ParameterInjectionMiddleware(
        params={"api_key": "trusted-key"},
        enforce={"api_key"},
    )
    tool = FunctionTool.from_defaults(query, middlewares=[mw])

    result = await tool.acall(text="hello", api_key="hacked-key")
    assert result.raw_output == "text=hello, key=trusted-key"


def test_middleware_via_direct_call() -> None:
    """Test that middleware is applied when using tool.call() directly."""

    def greet(text: str) -> str:
        return f"Hello, {text}"

    mw = AppendMiddleware("!")
    tool = FunctionTool.from_defaults(greet, middlewares=[mw])

    # call() directly — middleware should still run
    result = tool.call(text="world")
    assert result.raw_output == "Hello, world!!"


def test_output_filter_middleware_passthrough_non_dict() -> None:
    """Test OutputFilterMiddleware passes through non-dict/non-list output unchanged."""

    def get_message() -> str:
        return "just a plain string"

    mw = OutputFilterMiddleware(allowed_fields={"id"})
    tool = FunctionTool.from_defaults(get_message, middlewares=[mw])

    result = tool()
    assert result.raw_output == "just a plain string"


def test_parameter_injection_middleware_enforce_validation() -> None:
    """Test that enforce keys not in params raises ValueError."""
    with pytest.raises(ValueError, match="enforce keys"):
        ParameterInjectionMiddleware(
            params={"api_key": "key"},
            enforce={"api_key", "nonexistent"},
        )
