from typing import List, Optional

import pytest

from llama_index.core import MockEmbedding, VectorStoreIndex
from llama_index.core.schema import (
    NodeRelationship,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)


@pytest.fixture()
def document_nodes() -> List[TextNode]:
    return [
        TextNode(
            id_=f"{doc}-{chunk}",
            text=f"Document {doc}, chunk {chunk}",
            embedding=[1.0, float(chunk)],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc)},
            metadata={"chunk": chunk},
        )
        for doc in ["a", "b"]
        for chunk in [0, 1]
    ]


@pytest.mark.parametrize(
    ("doc_ids", "expected"),
    [
        (None, {"a-0", "a-1", "b-0", "b-1"}),
        ([], set()),
        (["missing"], set()),
        (["a"], {"a-0", "a-1"}),
    ],
)
@pytest.mark.parametrize(
    "mode", [VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.MMR]
)
def test_retriever_constrains_source_documents(
    document_nodes: List[TextNode],
    doc_ids: Optional[List[str]],
    expected: set[str],
    mode: VectorStoreQueryMode,
) -> None:
    index = VectorStoreIndex(document_nodes, embed_model=MockEmbedding(embed_dim=2))
    retriever = index.as_retriever(
        doc_ids=doc_ids, similarity_top_k=10, vector_store_query_mode=mode
    )

    results = retriever.retrieve(QueryBundle(query_str="test", embedding=[1.0, 0.0]))

    assert {item.node.node_id for item in results} == expected


def test_document_node_and_metadata_filters_intersect(
    document_nodes: List[TextNode],
) -> None:
    store = SimpleVectorStore()
    store.add(document_nodes)
    store.add(
        [
            TextNode(
                id_="a-2",
                embedding=[1.0, 0.0],
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="a")},
                metadata={"chunk": 1},
            )
        ]
    )
    result = store.query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0],
            similarity_top_k=10,
            doc_ids=["a"],
            node_ids=["a-0", "a-1", "b-1"],
            filters=MetadataFilters(filters=[ExactMatchFilter(key="chunk", value=1)]),
        )
    )
    assert result.ids == ["a-1"]


@pytest.mark.parametrize(
    ("doc_ids", "expected"), [(None, {"mapped", "unmapped"}), (["a"], {"mapped"})]
)
def test_document_filter_handles_missing_source_mapping(
    doc_ids: Optional[List[str]], expected: set[str]
) -> None:
    store = SimpleVectorStore()
    store.data.embedding_dict = {"mapped": [1.0, 0.0], "unmapped": [0.0, 1.0]}
    store.data.text_id_to_ref_doc_id = {"mapped": "a"}

    result = store.query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0], similarity_top_k=10, doc_ids=doc_ids
        )
    )

    assert set(result.ids) == expected


@pytest.mark.parametrize("doc_ids", [[], ["missing"]])
@pytest.mark.parametrize(
    "mode",
    [
        VectorStoreQueryMode.SVM,
        VectorStoreQueryMode.LINEAR_REGRESSION,
        VectorStoreQueryMode.LOGISTIC_REGRESSION,
    ],
)
def test_document_filter_with_no_candidates_in_learner_modes(
    document_nodes: List[TextNode],
    doc_ids: List[str],
    mode: VectorStoreQueryMode,
) -> None:
    store = SimpleVectorStore()
    store.add(document_nodes)

    result = store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0], doc_ids=doc_ids, mode=mode)
    )

    assert result.ids == []
    assert result.similarities == []
