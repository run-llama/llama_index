from typing import List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    RelatedNodeInfo,
)
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

TEXT = " ".join(
    f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(400)
)


class StubRetriever(BaseRetriever):
    """Retriever returning a fixed, ordered list of nodes without a vector store."""

    def __init__(self, results: List[NodeWithScore]) -> None:
        self._results = results
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._results


def _build_context(nodes: List[BaseNode]) -> StorageContext:
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    return StorageContext.from_defaults(docstore=docstore)


def _retriever(
    nodes: List[BaseNode], picked: List[BaseNode], threshold: float = 0.5
) -> AutoMergingRetriever:
    context = _build_context(nodes)
    stored = [context.docstore.get_document(n.node_id) for n in picked]
    return AutoMergingRetriever(
        vector_retriever=StubRetriever(
            [
                NodeWithScore(node=stored[0], score=0.9),
                NodeWithScore(node=stored[1], score=0.8),
            ]
        ),
        storage_context=context,
        simple_ratio_thresh=threshold,
    )


def _flat_nodes(metadata: dict) -> List[BaseNode]:
    return SentenceSplitter(chunk_size=64, chunk_overlap=0).get_nodes_from_documents(
        [Document(text=TEXT, id_="doc-1", metadata=metadata)]
    )


def test_fills_missing_neighbour_when_document_has_metadata() -> None:
    # Documents read with SimpleDirectoryReader carry metadata. The node parser builds
    # the NEXT reference (A -> B) before B inherits document metadata and the PREVIOUS
    # reference (B <- C) afterwards, so the two RelatedNodeInfo snapshots for B differ
    # in metadata/hash. Comparing them by value never fills in B.
    nodes = _flat_nodes({"file_name": "report.pdf"})
    a, b, c = nodes[3], nodes[4], nodes[5]

    results = _retriever(nodes, [a, c]).retrieve("query")

    assert [node.node.node_id for node in results] == [a.node_id, b.node_id, c.node_id]


def test_still_fills_missing_neighbour_without_metadata() -> None:
    nodes = _flat_nodes({})
    a, b, c = nodes[3], nodes[4], nodes[5]

    results = _retriever(nodes, [a, c]).retrieve("query")

    assert b.node_id in {node.node.node_id for node in results}


def test_uses_node_id_when_related_snapshots_disagree_in_metadata() -> None:
    # Hand-built references where the two snapshots point at the same node id but
    # carry different metadata/hash, as can happen with custom ingestion that edits
    # nodes after the parser ran. Fill-in must follow the node id.
    nodes = _flat_nodes({})
    a, b, c = nodes[3], nodes[4], nodes[5]
    context = _build_context(nodes)
    a_stored, c_stored = (
        context.docstore.get_document(a.node_id),
        context.docstore.get_document(c.node_id),
    )
    a_stored.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
        node_id=b.node_id, metadata={}, hash="stale-next"
    )
    c_stored.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
        node_id=b.node_id,
        metadata={"file_name": "report.pdf"},
        hash="stale-prev",
    )

    retriever = AutoMergingRetriever(
        vector_retriever=StubRetriever(
            [
                NodeWithScore(node=a_stored, score=0.9),
                NodeWithScore(node=c_stored, score=0.8),
            ]
        ),
        storage_context=context,
    )
    results = retriever.retrieve("query")

    assert b.node_id in {node.node.node_id for node in results}


def test_fill_in_does_not_reinsert_an_already_present_neighbour() -> None:
    # When the missing neighbour is itself a retrieved hit (retrieval order is by
    # similarity, not by position), _fill_in_nodes must not append it a second time.
    nodes = _flat_nodes({})
    a, b, c = nodes[3], nodes[4], nodes[5]
    context = _build_context(nodes)
    stored = [context.docstore.get_document(n.node_id) for n in (a, c, b)]
    input_nodes = [
        NodeWithScore(node=stored[0], score=0.9),
        NodeWithScore(node=stored[1], score=0.8),
        NodeWithScore(node=stored[2], score=0.7),
    ]
    retriever = AutoMergingRetriever(
        vector_retriever=StubRetriever(input_nodes), storage_context=context
    )

    filled, changed = retriever._fill_in_nodes(input_nodes)

    ids = [node.node.node_id for node in filled]
    assert changed is False
    assert len(ids) == len(set(ids))
    assert sorted(ids) == sorted([a.node_id, b.node_id, c.node_id])


def test_leaves_non_adjacent_retrieved_nodes_unchanged() -> None:
    # Two retrieved nodes with more than one node between them are not filled in.
    nodes = _flat_nodes({})
    a, c = nodes[3], nodes[6]

    results = _retriever(nodes, [a, c]).retrieve("query")

    assert [node.node.node_id for node in results] == [a.node_id, c.node_id]
