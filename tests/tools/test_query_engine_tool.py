"""Test ad-hoc loader Tool."""

from typing import List
from llama_index.readers.schema.base import Document
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.tools.query_engine import QueryEngineTool


def test_query_engine_tool_openai_json(
    mock_service_context: ServiceContext,
    documents: List[Document],
) -> None:
    """Test QueryEngineTool."""

    index = VectorStoreIndex.from_documents(documents)

    name = ("my vector index",)
    description = "my description"

    tool = QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(), name=name, description=description
    )

    # create tool json
    openai_json = tool.as_openai_json_function()

    # verify schema
    assert "name" in openai_json.keys()
    assert openai_json["name"] == name

    assert "description" in openai_json.keys()
    assert openai_json["description"] == description

    assert "parameters" in openai_json.keys()
    assert "properties" in openai_json["parameters"].keys()

    # should only have one function
    assert len(openai_json["parameters"]["properties"]) == 1
