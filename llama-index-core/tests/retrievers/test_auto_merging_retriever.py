from typing import List
from unittest.mock import MagicMock

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.storage.storage_context import StorageContext


def _build_three_chunk_storage() -> tuple[StorageContext, List[TextNode]]:
    n0 = TextNode(id_="chunk-0", text="Price is 4 USD per pack.")
    n1 = TextNode(id_="chunk-1", text="Minimum order of ten packs.")
    n2 = TextNode(id_="chunk-2", text="Submit order to the desk.")

    n0.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=n1.node_id)
    n1.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=n0.node_id)
    n1.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=n2.node_id)
    n2.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=n1.node_id)

    storage = StorageContext.from_defaults()
    storage.docstore.add_documents([n0, n1, n2])
    return storage, [n0, n1, n2]


def test_auto_merging_fills_gap_when_right_hit_ranks_first() -> None:
    storage, (n0, n1, n2) = _build_three_chunk_storage()
    base_retriever = MagicMock()
    base_retriever.retrieve.return_value = [
        NodeWithScore(node=n2, score=0.9),
        NodeWithScore(node=n0, score=0.7),
    ]

    retriever = AutoMergingRetriever(base_retriever, storage)
    results = retriever.retrieve(QueryBundle("Can I buy eight packs?"))

    assert [r.node.node_id for r in results] == ["chunk-2", "chunk-1", "chunk-0"]
    assert results[1].get_score() == (0.9 + 0.7) / 2


def test_auto_merging_fills_gap_across_non_adjacent_hits_without_duplicating() -> None:
    storage, (n0, n1, n2) = _build_three_chunk_storage()
    unrelated = TextNode(id_="other", text="Unrelated policy.")
    storage.docstore.add_documents([unrelated])

    base_retriever = MagicMock()
    base_retriever.retrieve.return_value = [
        NodeWithScore(node=n0, score=0.8),
        NodeWithScore(node=unrelated, score=0.75),
        NodeWithScore(node=n2, score=0.6),
    ]

    retriever = AutoMergingRetriever(base_retriever, storage)
    results = retriever.retrieve(QueryBundle("policy"))

    ids = [r.node.node_id for r in results]
    assert set(ids) == {"chunk-0", "chunk-1", "chunk-2", "other"}
    assert len(ids) == 4
