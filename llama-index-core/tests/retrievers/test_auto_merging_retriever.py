from typing import List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.storage.storage_context import StorageContext


class _FixedRetriever(BaseRetriever):
    def __init__(self, nodes: List[NodeWithScore]) -> None:
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return list(self._nodes)


def _link_siblings(nodes: List[TextNode]) -> None:
    for idx, node in enumerate(nodes):
        if idx > 0:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[idx - 1].node_id
            )
        if idx < len(nodes) - 1:
            node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=nodes[idx + 1].node_id
            )


def _attach_parent(parent: TextNode, children: List[TextNode]) -> None:
    parent.relationships[NodeRelationship.CHILD] = [
        RelatedNodeInfo(node_id=child.node_id) for child in children
    ]
    for child in children:
        child.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
            node_id=parent.node_id
        )


def test_fill_in_and_merge_into_parent() -> None:
    parent = TextNode(text="c1 c2 c3 c4", id_="parent")
    children = [TextNode(text=f"c{i}", id_=f"c{i}") for i in range(1, 5)]
    _link_siblings(children)
    _attach_parent(parent, children)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents([parent, *children])

    # Retrieve c1 and c3; c2 should be filled in -> 3/4 = 0.75 > 0.5 -> merges into parent
    initial = [
        NodeWithScore(node=children[0], score=0.9),
        NodeWithScore(node=children[2], score=0.7),
    ]
    retriever = AutoMergingRetriever(
        vector_retriever=_FixedRetriever(initial),  # type: ignore[arg-type]
        storage_context=storage_context,
        simple_ratio_thresh=0.5,
    )

    result = retriever.retrieve("query")
    assert [n.node.node_id for n in result] == ["parent"]
    # c1 (0.9), c3 (0.7), filled c2 (0.8) -> average = 0.8
    assert abs((result[0].score or 0.0) - 0.8) < 1e-6


def test_fill_in_nodes_skips_already_present_gap_node() -> None:
    parent = TextNode(text="full parent", id_="parent")
    children = [TextNode(text=f"c{i}", id_=f"c{i}") for i in range(1, 7)]
    _link_siblings(children)
    _attach_parent(parent, children)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents([parent, *children])

    # Vector search returns c1, c3, c2 ordered by similarity score.
    # Unique children = 3/6 = 0.5 <= 0.5 (should NOT merge into parent, and c2 must not be duplicated).
    initial = [
        NodeWithScore(node=children[0], score=0.95),
        NodeWithScore(node=children[2], score=0.85),
        NodeWithScore(node=children[1], score=0.75),
    ]
    retriever = AutoMergingRetriever(
        vector_retriever=_FixedRetriever(initial),  # type: ignore[arg-type]
        storage_context=storage_context,
        simple_ratio_thresh=0.5,
    )

    result = retriever.retrieve("query")
    assert [n.node.node_id for n in result] == ["c1", "c3", "c2"]


def test_retrieve_terminates_when_filled_gap_node_merges_into_parent() -> None:
    p1 = TextNode(text="p1", id_="p1")
    p2 = TextNode(text="p2", id_="p2")
    p3 = TextNode(text="p3", id_="p3")

    p1_children = [TextNode(text=f"p1_c{i}", id_=f"p1_c{i}") for i in range(1, 5)]
    c2 = TextNode(text="p2_c1", id_="c2")
    p3_children = [TextNode(text=f"p3_c{i}", id_=f"p3_c{i}") for i in range(1, 5)]

    c1 = p1_children[-1]
    c3 = p3_children[0]
    _link_siblings([*p1_children, c2, *p3_children])
    _attach_parent(p1, p1_children)
    _attach_parent(p2, [c2])
    _attach_parent(p3, p3_children)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(
        [p1, p2, p3, *p1_children, c2, *p3_children]
    )

    # c1 and c3 flank c2. Once c2 is filled in and merged into p2,
    # c1 and c3 become adjacent again; _retrieve must not re-fill c2 in an infinite loop.
    initial = [
        NodeWithScore(node=c1, score=0.9),
        NodeWithScore(node=c3, score=0.7),
    ]
    retriever = AutoMergingRetriever(
        vector_retriever=_FixedRetriever(initial),  # type: ignore[arg-type]
        storage_context=storage_context,
        simple_ratio_thresh=0.5,
    )

    result = retriever.retrieve("query")
    assert [n.node.node_id for n in result] == ["p1_c4", "p2", "p3_c1"]


def test_get_parents_and_merge_deduplicates_children_by_node_id() -> None:
    parent = TextNode(text="parent", id_="parent")
    children = [TextNode(text=f"c{i}", id_=f"c{i}") for i in range(1, 4)]
    _attach_parent(parent, children)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents([parent, *children])

    # Duplicate c1 should count as 1 child out of 3 (ratio 1/3 <= 0.5), not 2/3.
    initial = [
        NodeWithScore(node=children[0], score=0.9),
        NodeWithScore(node=children[0], score=0.9),
    ]
    retriever = AutoMergingRetriever(
        vector_retriever=_FixedRetriever(initial),  # type: ignore[arg-type]
        storage_context=storage_context,
        simple_ratio_thresh=0.5,
    )

    merged, is_changed = retriever._get_parents_and_merge(initial)
    assert not is_changed
    assert all(n.node.node_id != "parent" for n in merged)


def test_get_parents_and_merge_does_not_duplicate_existing_parent() -> None:
    grandparent = TextNode(text="gp", id_="gp")
    p1 = TextNode(text="p1", id_="p1")
    p2 = TextNode(text="p2", id_="p2")
    p3 = TextNode(text="p3", id_="p3")
    _attach_parent(grandparent, [p1, p2, p3])

    c1 = TextNode(text="c1", id_="c1")
    c2 = TextNode(text="c2", id_="c2")
    _attach_parent(p1, [c1, c2])

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents([grandparent, p1, p2, p3, c1, c2])

    # p1 is already present alongside its children c1 and c2.
    # Merging c1 and c2 into p1 must yield a single p1 (1/3 of gp <= 0.5, so gp must NOT merge).
    initial = [
        NodeWithScore(node=p1, score=0.8),
        NodeWithScore(node=c1, score=0.9),
        NodeWithScore(node=c2, score=0.7),
    ]
    retriever = AutoMergingRetriever(
        vector_retriever=_FixedRetriever(initial),  # type: ignore[arg-type]
        storage_context=storage_context,
        simple_ratio_thresh=0.5,
    )

    result = retriever.retrieve("query")
    assert [n.node.node_id for n in result] == ["p1"]


def test_retrieve_and_fill_in_nodes_handle_none_scores() -> None:
    parent = TextNode(text="parent", id_="parent")
    children = [TextNode(text=f"c{i}", id_=f"c{i}") for i in range(1, 4)]
    _link_siblings(children)
    _attach_parent(parent, children)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents([parent, *children])

    initial = [
        NodeWithScore(node=children[0], score=None),
        NodeWithScore(node=children[2], score=None),
    ]
    retriever = AutoMergingRetriever(
        vector_retriever=_FixedRetriever(initial),  # type: ignore[arg-type]
        storage_context=storage_context,
        simple_ratio_thresh=0.5,
    )

    result = retriever.retrieve("query")
    assert [n.node.node_id for n in result] == ["parent"]
    assert result[0].score == 0.0
