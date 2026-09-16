import json
from typing import Any, Dict, Iterator
from unittest.mock import patch
from uuid import UUID

import pytest
from google.cloud.discoveryengine_v1beta import Document, SearchResponse
from google.protobuf.json_format import MessageToDict

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.retrievers.vertexai_search.base import VertexAISearchRetriever


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in VertexAISearchRetriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes


@pytest.fixture
def structured_retriever() -> Iterator[VertexAISearchRetriever]:
    with patch("google.cloud.discoveryengine_v1beta.SearchServiceClient") as client:
        client.return_value.serving_config_path.return_value = (
            "projects/project/locations/global/dataStores/store/servingConfigs/default"
        )
        yield VertexAISearchRetriever(
            project_id="project", data_store_id="store", engine_data_type=1
        )


def test_structured_retrieve_preserves_metadata_and_generated_node_ids(
    structured_retriever: VertexAISearchRetriever,
) -> None:
    struct_data = {
        "title": "Café",
        "source_uri": "gs://bucket/document.json",
        "tenant_id": "tenant-123",
        "expected_doc_id": "expected-id",
        "price": 12.5,
        "available": True,
        "optional": None,
        "details": {"categories": ["food", "drink"], "count": 2},
    }
    results = [
        SearchResponse.SearchResult(
            document=Document(
                id=document_id,
                name=f"projects/project/locations/global/dataStores/store/branches/0/documents/{document_id}",
                struct_data=struct_data,
            )
        )
        for document_id in ("first", "second")
    ]
    response = SearchResponse(results=results)
    structured_retriever._client.search.return_value = response

    nodes = structured_retriever.retrieve("café")
    repeated_nodes = structured_retriever.retrieve("café")

    assert len(nodes) == len(repeated_nodes) == 2
    assert [node.score for node in nodes] == [1.0, 0.5]
    for result, node, repeated_node in zip(response.results, nodes, repeated_nodes):
        expected_metadata = {
            **struct_data,
            "document_id": result.document.id,
            "document_name": result.document.name,
        }
        expected_text = json.dumps(
            MessageToDict(result.document._pb, preserving_proto_field_name=True)[
                "struct_data"
            ]
        )
        assert node.metadata == repeated_node.metadata == expected_metadata
        assert node.node.text == repeated_node.node.text == expected_text
        assert node.get_content() == expected_text

    node_ids = [node.node.id_ for node in nodes + repeated_nodes]
    assert len(set(node_ids)) == 4
    assert all(UUID(node_id).version == 4 for node_id in node_ids)


@pytest.mark.parametrize(
    ("document", "expected_text", "expected_metadata"),
    [
        (Document(), "{}", {"document_id": "", "document_name": ""}),
        (
            Document(id="document-id", name="document-name"),
            "{}",
            {"document_id": "document-id", "document_name": "document-name"},
        ),
        (
            Document(struct_data={"title": "Example"}),
            '{"title": "Example"}',
            {"title": "Example", "document_id": "", "document_name": ""},
        ),
    ],
)
def test_structured_response_missing_fields(
    structured_retriever: VertexAISearchRetriever,
    document: Document,
    expected_text: str,
    expected_metadata: Dict[str, Any],
) -> None:
    nodes = structured_retriever._convert_structured_datastore_response(
        [SearchResponse.SearchResult(document=document)]
    )

    assert len(nodes) == 1
    assert nodes[0].node.text == expected_text
    assert nodes[0].metadata == expected_metadata
    assert nodes[0].score == 1.0


@pytest.mark.parametrize(
    "identity", [{"id": "canonical-id", "name": "canonical-name"}, {}]
)
def test_structured_metadata_prefers_document_identity(
    structured_retriever: VertexAISearchRetriever, identity: Dict[str, str]
) -> None:
    struct_data = {
        "document_id": "source-id",
        "document_name": "source-name",
        "title": "Example",
    }
    document = Document(struct_data=struct_data, **identity)
    result = SearchResponse.SearchResult(document=document)

    nodes = structured_retriever._convert_structured_datastore_response([result])

    assert nodes[0].metadata == {
        "document_id": identity.get("id", ""),
        "document_name": identity.get("name", ""),
        "title": "Example",
    }
    assert json.loads(nodes[0].node.text) == struct_data
    assert nodes[0].node.text == json.dumps(
        MessageToDict(result.document._pb, preserving_proto_field_name=True)[
            "struct_data"
        ]
    )


def test_structured_response_empty(
    structured_retriever: VertexAISearchRetriever,
) -> None:
    assert structured_retriever._convert_structured_datastore_response([]) == []
