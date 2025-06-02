"""Test load and search tool spec."""

from typing import Any, List
import pytest
from unittest.mock import MagicMock, patch

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_index.core.tools.types import ToolMetadata


class TestSchema(BaseModel):
    query: str


def _foo(query: str) -> List[Document]:
    return [Document(text=f"Test document with query: {query}")]


def test_load_and_search_tool_spec_init() -> None:
    function_tool = FunctionTool.from_defaults(
        fn=_foo,
        name="test_loader",
        description="Test loader function",
        fn_schema=TestSchema,
    )

    metadata = ToolMetadata(
        name="test_loader",
        description="Test loader function",
        fn_schema=TestSchema,
    )

    tool_spec = LoadAndSearchToolSpec(
        tool=function_tool,
        index_cls=VectorStoreIndex,
        index_kwargs={},
        metadata=metadata,
    )

    assert tool_spec.metadata.name == "test_loader"
    assert tool_spec.metadata.description == "Test loader function"
    assert tool_spec.metadata.fn_schema == TestSchema

    assert tool_spec.spec_functions == ["test_loader", "read_test_loader"]

    tools = tool_spec.to_tool_list()
    assert len(tools) == 2
    assert tools[0].metadata.name == "test_loader"
    assert "test_loader" in tools[0].metadata.description
    assert tools[1].metadata.name == "read_test_loader"
    assert "read" in tools[1].metadata.description


def test_load_and_search_tool_spec_from_defaults() -> None:
    function_tool = FunctionTool.from_defaults(
        fn=_foo,
        name="test_loader",
        description="Test loader function",
        fn_schema=TestSchema,
    )

    tool_spec = LoadAndSearchToolSpec.from_defaults(
        tool=function_tool,
    )

    assert tool_spec.metadata.name == "test_loader"
    assert tool_spec.metadata.description == "Test loader function"
    assert tool_spec.metadata.fn_schema == TestSchema

    tool_spec = LoadAndSearchToolSpec.from_defaults(
        tool=function_tool,
        name="custom_name",
        description="Custom description",
    )

    assert tool_spec.metadata.name == "custom_name"
    assert tool_spec.metadata.description == "Custom description"


def test_load() -> None:
    mock_function = MagicMock(return_value=MagicMock(raw_output="Test document"))
    function_tool = FunctionTool.from_defaults(
        fn=mock_function,
        name="test_loader",
        description="Test loader function",
    )

    tool_spec = LoadAndSearchToolSpec.from_defaults(
        tool=function_tool,
    )

    result = tool_spec.load(query="input query")
    assert "Content loaded!" in result
    assert "read_test_loader" in result
    assert tool_spec._index is not None

    tool_spec._index = None

    mock_function.return_value = MagicMock(raw_output=Document(text="Test document"))
    result = tool_spec.load(query="input query")
    assert "Content loaded!" in result
    assert tool_spec._index is not None

    tool_spec._index = None

    mock_function.return_value = MagicMock(raw_output=["Doc1", "Doc2"])
    result = tool_spec.load(query="input query")
    assert "Content loaded!" in result
    assert tool_spec._index is not None

    tool_spec._index = None

    mock_function.return_value = MagicMock(
        raw_output=[Document(text="Doc1"), Document(text="Doc2")]
    )
    result = tool_spec.load(query="input query")
    assert "Content loaded!" in result
    assert tool_spec._index is not None

    mock_index = MagicMock()
    tool_spec._index = mock_index
    mock_function.return_value = MagicMock(raw_output="New document")
    result = tool_spec.load(query="input query")
    assert "Content loaded!" in result
    assert mock_index.insert.called


def test_read() -> None:
    function_tool = FunctionTool.from_defaults(
        fn=_foo,
        name="test_loader",
        description="Test loader function",
    )

    tool_spec = LoadAndSearchToolSpec.from_defaults(
        tool=function_tool,
    )

    result = tool_spec.read(query="input query")
    assert "Error" in result
    assert "No content has been loaded" in result
    assert "test_loader" in result

    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = "Query result"

    mock_index = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine

    tool_spec._index = mock_index

    result = tool_spec.read(query="input query")
    assert result == "Query result"
    mock_query_engine.query.assert_called_once_with("input query")


@pytest.mark.parametrize(
    ("raw_output", "expected_doc_count"),
    [
        ("Single string", 1),
        (Document(text="Single document"), 1),
        (123, 1),
    ],
)
def test_load_different_output_types(raw_output: Any, expected_doc_count: int) -> None:
    mock_function = MagicMock(return_value=MagicMock(raw_output=raw_output))
    function_tool = FunctionTool.from_defaults(
        fn=mock_function,
        name="test_loader",
        description="Test loader function",
    )

    mock_index = MagicMock()
    mock_index_cls = MagicMock(return_value=mock_index)
    mock_index_cls.from_documents = MagicMock(return_value=mock_index)

    tool_spec = LoadAndSearchToolSpec(
        tool=function_tool,
        index_cls=mock_index_cls,
        index_kwargs={},
        metadata=ToolMetadata(name="test_loader", description="Test loader"),
    )

    tool_spec.load(query="input query")

    args, _ = mock_index_cls.from_documents.call_args
    assert len(args[0]) == expected_doc_count


def test_load_edge_cases() -> None:
    def custom_string_function(*args, **kwargs):
        return "Single string"

    function_tool = FunctionTool.from_defaults(
        fn=custom_string_function,
        name="test_loader",
        description="Test loader function",
    )

    mock_index = MagicMock()

    with patch(
        "llama_index.core.indices.vector_store.VectorStoreIndex.from_documents",
        return_value=mock_index,
    ) as mock_from_docs:
        tool_spec = LoadAndSearchToolSpec.from_defaults(
            tool=function_tool,
            index_cls=VectorStoreIndex,
            index_kwargs={},
        )

        tool_spec.load(query="input query")

        mock_from_docs.assert_called_once()
        docs_arg = mock_from_docs.call_args[0][0]

        assert len(docs_arg) == 1
        assert isinstance(docs_arg[0], Document)
        assert docs_arg[0].text == "Single string"

    doc = Document(text="Single document")

    def custom_doc_function(*args, **kwargs):
        return doc

    function_tool = FunctionTool.from_defaults(
        fn=custom_doc_function,
        name="test_loader",
        description="Test loader function",
    )

    with patch(
        "llama_index.core.indices.vector_store.VectorStoreIndex.from_documents",
        return_value=mock_index,
    ) as mock_from_docs:
        tool_spec = LoadAndSearchToolSpec.from_defaults(
            tool=function_tool,
            index_cls=VectorStoreIndex,
            index_kwargs={},
        )

        tool_spec.load(query="input query")

        mock_from_docs.assert_called_once()
        docs_arg = mock_from_docs.call_args[0][0]

        assert len(docs_arg) == 1
        assert docs_arg[0] == doc

    with pytest.raises(ValueError, match="Tool name cannot be None"):
        LoadAndSearchToolSpec(
            tool=function_tool,
            index_cls=VectorStoreIndex,
            index_kwargs={},
            metadata=ToolMetadata(name=None, description="Test loader"),
        )


def test_load_list_output_types() -> None:
    def custom_function(*args, **kwargs):
        return ["String 1", "String 2"]

    function_tool = FunctionTool.from_defaults(
        fn=custom_function,
        name="test_loader",
        description="Test loader function",
    )

    mock_index = MagicMock()

    with patch(
        "llama_index.core.indices.vector_store.VectorStoreIndex.from_documents",
        return_value=mock_index,
    ) as mock_from_docs:
        tool_spec = LoadAndSearchToolSpec.from_defaults(
            tool=function_tool,
            index_cls=VectorStoreIndex,
            index_kwargs={},
        )

        tool_spec.load(query="input query")

        mock_from_docs.assert_called_once()
        docs_arg = mock_from_docs.call_args[0][0]

        assert len(docs_arg) == 2
        assert isinstance(docs_arg[0], Document)
        assert isinstance(docs_arg[1], Document)
        assert docs_arg[0].text == "String 1"
        assert docs_arg[1].text == "String 2"

    def custom_doc_function(*args, **kwargs):
        return [Document(text="Doc 1"), Document(text="Doc 2")]

    function_tool = FunctionTool.from_defaults(
        fn=custom_doc_function,
        name="test_loader",
        description="Test loader function",
    )

    with patch(
        "llama_index.core.indices.vector_store.VectorStoreIndex.from_documents",
        return_value=mock_index,
    ) as mock_from_docs:
        tool_spec = LoadAndSearchToolSpec.from_defaults(
            tool=function_tool,
            index_cls=VectorStoreIndex,
            index_kwargs={},
        )

        tool_spec.load(query="input query")

        mock_from_docs.assert_called_once()
        docs_arg = mock_from_docs.call_args[0][0]

        assert len(docs_arg) == 2
        assert isinstance(docs_arg[0], Document)
        assert isinstance(docs_arg[1], Document)
        assert docs_arg[0].text == "Doc 1"
        assert docs_arg[1].text == "Doc 2"
