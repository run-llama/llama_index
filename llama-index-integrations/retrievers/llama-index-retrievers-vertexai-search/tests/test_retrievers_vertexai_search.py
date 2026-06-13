import json
from types import SimpleNamespace

from google.cloud.discoveryengine_v1.types import Document, SearchResponse
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.retrievers.vertexai_search.base import VertexAISearchRetriever


def test_class():
    names_of_base_classes = [b.__name__ for b in VertexAISearchRetriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes


def test_structured_response_preserves_document_identity_and_metadata():
    """
    Regression test for #21933.

    Structured data store results must keep the Discovery Engine document identity
    (`id`/`name`) and the `struct_data` fields, instead of only serializing the struct
    payload into the node text.
    """
    document = Document(
        id="li-meta-gamma",
        name=(
            "projects/PN/locations/global/collections/default_collection/"
            "dataStores/DS/branches/0/documents/li-meta-gamma"
        ),
        struct_data={
            "case_id": "RUN",
            "title": "Gamma",
            "expected_doc_id": "li-meta-gamma",
        },
    )
    result = SearchResponse.SearchResult(id="li-meta-gamma", document=document)

    nodes = VertexAISearchRetriever._convert_structured_datastore_response(
        SimpleNamespace(), [result]
    )

    assert len(nodes) == 1
    node = nodes[0].node
    assert node.id_ == "li-meta-gamma"
    assert node.metadata["case_id"] == "RUN"
    assert node.metadata["title"] == "Gamma"
    assert node.metadata["id"] == "li-meta-gamma"
    assert node.metadata["name"].endswith("documents/li-meta-gamma")
    assert json.loads(node.text)["case_id"] == "RUN"
