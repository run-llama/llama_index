"""Node postprocessor tests."""

from importlib.util import find_spec
from pathlib import Path
from typing import Dict, cast

import pytest
from llama_index.core.postprocessor.node import (
    KeywordNodePostprocessor,
    PrevNextNodePostprocessor,
)
from llama_index.core.postprocessor.node_recency import (
    EmbeddingRecencyPostprocessor,
    FixedRecencyPostprocessor,
    TimeWeightedPostprocessor,
)
from llama_index.core.schema import (
    MetadataMode,
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore

spacy_installed = bool(find_spec("spacy"))


def test_forward_back_processor(tmp_path: Path) -> None:
    """Test forward-back processor."""
    nodes = [
        TextNode(text="Hello world.", id_="3"),
        TextNode(text="This is a test.", id_="2"),
        TextNode(text="This is another test.", id_="1"),
        TextNode(text="This is a test v2.", id_="4"),
        TextNode(text="This is a test v3.", id_="5"),
    ]
    nodes_with_scores = [NodeWithScore(node=node) for node in nodes]
    for i, node in enumerate(nodes):
        if i > 0:
            node.relationships.update(
                {
                    NodeRelationship.PREVIOUS: RelatedNodeInfo(
                        node_id=nodes[i - 1].node_id
                    )
                },
            )
        if i < len(nodes) - 1:
            node.relationships.update(
                {NodeRelationship.NEXT: RelatedNodeInfo(node_id=nodes[i + 1].node_id)},
            )

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # check for a single node
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=2, mode="next"
    )
    processed_nodes = node_postprocessor.postprocess_nodes([nodes_with_scores[0]])
    assert len(processed_nodes) == 3
    assert processed_nodes[0].node.node_id == "3"
    assert processed_nodes[1].node.node_id == "2"
    assert processed_nodes[2].node.node_id == "1"

    # check for multiple nodes (nodes should not be duped)
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=1, mode="next"
    )
    processed_nodes = node_postprocessor.postprocess_nodes(
        [
            nodes_with_scores[1],
            nodes_with_scores[2],
        ]
    )
    assert len(processed_nodes) == 3
    assert processed_nodes[0].node.node_id == "2"
    assert processed_nodes[1].node.node_id == "1"
    assert processed_nodes[2].node.node_id == "4"

    # check for previous
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=1, mode="previous"
    )
    processed_nodes = node_postprocessor.postprocess_nodes(
        [
            nodes_with_scores[1],
            nodes_with_scores[2],
        ]
    )
    assert len(processed_nodes) == 3
    assert processed_nodes[0].node.node_id == "3"
    assert processed_nodes[1].node.node_id == "2"
    assert processed_nodes[2].node.node_id == "1"

    # check that both works
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=1, mode="both"
    )
    processed_nodes = node_postprocessor.postprocess_nodes([nodes_with_scores[2]])
    assert len(processed_nodes) == 3
    # nodes are sorted
    assert processed_nodes[0].node.node_id == "2"
    assert processed_nodes[1].node.node_id == "1"
    assert processed_nodes[2].node.node_id == "4"

    # check that num_nodes too high still works
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=4, mode="both"
    )
    processed_nodes = node_postprocessor.postprocess_nodes([nodes_with_scores[2]])
    assert len(processed_nodes) == 5
    # nodes are sorted
    assert processed_nodes[0].node.node_id == "3"
    assert processed_nodes[1].node.node_id == "2"
    assert processed_nodes[2].node.node_id == "1"
    assert processed_nodes[3].node.node_id == "4"
    assert processed_nodes[4].node.node_id == "5"

    # check that nodes with gaps works
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=1, mode="both"
    )
    processed_nodes = node_postprocessor.postprocess_nodes(
        [
            nodes_with_scores[0],
            nodes_with_scores[4],
        ]
    )
    assert len(processed_nodes) == 4
    # nodes are sorted
    assert processed_nodes[0].node.node_id == "3"
    assert processed_nodes[1].node.node_id == "2"
    assert processed_nodes[2].node.node_id == "4"
    assert processed_nodes[3].node.node_id == "5"

    # check that nodes with gaps works
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=0, mode="both"
    )
    processed_nodes = node_postprocessor.postprocess_nodes(
        [
            nodes_with_scores[0],
            nodes_with_scores[4],
        ]
    )
    assert len(processed_nodes) == 2
    # nodes are sorted
    assert processed_nodes[0].node.node_id == "3"
    assert processed_nodes[1].node.node_id == "5"

    # check that raises value error for invalid mode
    with pytest.raises(ValueError):
        PrevNextNodePostprocessor(docstore=docstore, num_nodes=4, mode="asdfasdf")


def _linked_text_nodes(node_ids: list[str]) -> list[TextNode]:
    nodes = [
        TextNode(text=f"Section {i}", id_=node_id) for i, node_id in enumerate(node_ids)
    ]
    for i, node in enumerate(nodes):
        if i > 0:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[i - 1].node_id
            )
        if i + 1 < len(nodes):
            node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=nodes[i + 1].node_id
            )
    return nodes


@pytest.mark.parametrize(
    ("mode", "retrieved_indices", "expected_indices"),
    [
        ("both", [3, 0], [0, 1, 2, 3, 4]),
        ("both", [0, 3], [0, 1, 2, 3, 4]),
        ("next", [2, 0], [0, 1, 2, 3]),
        ("previous", [3, 0, 2], [0, 1, 2, 3]),
        ("both", [3, 1, 3], [0, 1, 2, 3, 4]),
        ("both", [], []),
    ],
)
def test_prev_next_orders_joined_windows(
    mode: str, retrieved_indices: list[int], expected_indices: list[int]
) -> None:
    nodes = _linked_text_nodes(["z", "a", "x", "b", "y"])
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    postprocessor = PrevNextNodePostprocessor(docstore=docstore, num_nodes=1, mode=mode)

    processed = postprocessor.postprocess_nodes(
        [NodeWithScore(node=nodes[i], score=1.0) for i in retrieved_indices]
    )

    assert [node.node_id for node in processed] == [
        nodes[i].node_id for i in expected_indices
    ]


@pytest.mark.parametrize(
    "relationship", [NodeRelationship.PREVIOUS, NodeRelationship.NEXT]
)
def test_prev_next_orders_one_sided_relationships(
    relationship: NodeRelationship,
) -> None:
    nodes = _linked_text_nodes(["z", "a", "x", "b"])
    for node in nodes:
        node.relationships = {
            key: value
            for key, value in node.relationships.items()
            if key == relationship
        }
    postprocessor = PrevNextNodePostprocessor(
        docstore=SimpleDocumentStore(), num_nodes=0
    )

    processed = postprocessor.postprocess_nodes(
        [NodeWithScore(node=nodes[i]) for i in [3, 0, 1, 2]]
    )

    assert [node.node_id for node in processed] == [node.node_id for node in nodes]


def test_prev_next_reordering_preserves_scored_nodes() -> None:
    nodes = _linked_text_nodes(["z", "a", "x", "b"])
    scored_nodes = [
        NodeWithScore(node=node, score=score)
        for node, score in zip(nodes, [0.8, None, 0.0, 0.9])
    ]
    postprocessor = PrevNextNodePostprocessor(
        docstore=SimpleDocumentStore(), num_nodes=0
    )

    processed = postprocessor.postprocess_nodes([scored_nodes[i] for i in [3, 0, 1, 2]])

    assert [node.score for node in processed] == [0.8, None, 0.0, 0.9]
    assert all(actual is original for actual, original in zip(processed, scored_nodes))


def test_prev_next_preserves_disconnected_chain_order() -> None:
    nodes = _linked_text_nodes(list("ABCDEF"))
    other_nodes = _linked_text_nodes(["other-2", "other-1"])
    postprocessor = PrevNextNodePostprocessor(
        docstore=SimpleDocumentStore(), num_nodes=0
    )
    retrieved = [
        nodes[4],
        other_nodes[1],
        nodes[0],
        nodes[5],
        other_nodes[0],
        nodes[1],
    ]

    processed = postprocessor.postprocess_nodes(
        [NodeWithScore(node=node) for node in retrieved]
    )

    assert [node.node_id for node in processed] == [
        "E",
        "F",
        "other-2",
        "other-1",
        "A",
        "B",
    ]


@pytest.mark.parametrize("node_ids", [["self"], ["A", "B", "C"]])
def test_prev_next_cyclic_relationships_keep_each_node_once(
    node_ids: list[str],
) -> None:
    nodes = _linked_text_nodes(node_ids)
    nodes[0].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
        node_id=nodes[-1].node_id
    )
    nodes[-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
        node_id=nodes[0].node_id
    )
    postprocessor = PrevNextNodePostprocessor(
        docstore=SimpleDocumentStore(), num_nodes=0
    )

    processed = postprocessor.postprocess_nodes(
        [NodeWithScore(node=node) for node in nodes]
    )

    assert len(processed) == len(nodes)
    assert {node.node_id for node in processed} == set(node_ids)


@pytest.mark.parametrize("indices", [[0, 1, 2, 3], [3, 0, 1, 2]])
def test_prev_next_conflicting_relationships_keep_each_node_once(
    indices: list[int],
) -> None:
    nodes = _linked_text_nodes(list("ABC"))
    nodes.append(
        TextNode(
            id_="D",
            text="Conflicting neighbor",
            relationships={NodeRelationship.PREVIOUS: RelatedNodeInfo(node_id="B")},
        )
    )
    postprocessor = PrevNextNodePostprocessor(
        docstore=SimpleDocumentStore(), num_nodes=0
    )

    processed = postprocessor.postprocess_nodes(
        [NodeWithScore(node=nodes[i]) for i in indices]
    )

    assert len(processed) == len(nodes)
    assert {node.node_id for node in processed} == set("ABCD")


def test_fixed_recency_postprocessor() -> None:
    """Test fixed recency processor."""
    # try in metadata
    nodes = [
        TextNode(
            text="Hello world.",
            id_="1",
            metadata={"date": "2020-01-01"},
            excluded_embed_metadata_keys=["date"],
        ),
        TextNode(
            text="This is a test.",
            id_="2",
            metadata={"date": "2020-01-02"},
            excluded_embed_metadata_keys=["date"],
        ),
        TextNode(
            text="This is another test.",
            id_="3",
            metadata={"date": "2020-01-03"},
            excluded_embed_metadata_keys=["date"],
        ),
        TextNode(
            text="This is a test v2.",
            id_="4",
            metadata={"date": "2020-01-04"},
            excluded_embed_metadata_keys=["date"],
        ),
    ]
    node_with_scores = [NodeWithScore(node=node) for node in nodes]

    postprocessor = FixedRecencyPostprocessor(top_k=1)
    query_bundle: QueryBundle = QueryBundle(query_str="What is?")
    result_nodes = postprocessor.postprocess_nodes(
        node_with_scores, query_bundle=query_bundle
    )
    assert len(result_nodes) == 1
    assert (
        result_nodes[0].node.get_content(metadata_mode=MetadataMode.ALL)
        == "date: 2020-01-04\n\nThis is a test v2."
    )


def test_embedding_recency_postprocessor() -> None:
    """Test fixed recency processor."""
    # try in node info
    nodes = [
        TextNode(
            text="Hello world.",
            id_="1",
            metadata={"date": "2020-01-01"},
            excluded_embed_metadata_keys=["date"],
        ),
        TextNode(
            text="This is a test.",
            id_="2",
            metadata={"date": "2020-01-02"},
            excluded_embed_metadata_keys=["date"],
        ),
        TextNode(
            text="This is another test.",
            id_="3",
            metadata={"date": "2020-01-02"},
            excluded_embed_metadata_keys=["date"],
        ),
        TextNode(
            text="This is another test.",
            id_="3v2",
            metadata={"date": "2020-01-03"},
            excluded_embed_metadata_keys=["date"],
        ),
        TextNode(
            text="This is a test v2.",
            id_="4",
            metadata={"date": "2020-01-04"},
            excluded_embed_metadata_keys=["date"],
        ),
    ]
    nodes_with_scores = [NodeWithScore(node=node) for node in nodes]

    postprocessor = EmbeddingRecencyPostprocessor(
        top_k=1,
        in_metadata=False,
        query_embedding_tmpl="{context_str}",
    )
    query_bundle: QueryBundle = QueryBundle(query_str="What is?")
    result_nodes = postprocessor.postprocess_nodes(
        nodes_with_scores, query_bundle=query_bundle
    )
    # TODO: bring back this test
    # assert len(result_nodes) == 4
    assert result_nodes[0].node.get_content() == "This is a test v2."
    assert cast(Dict, result_nodes[0].node.metadata)["date"] == "2020-01-04"
    # assert result_nodes[1].node.get_content() == "This is another test."
    # assert result_nodes[1].node.node_id == "3v2"
    # assert cast(Dict, result_nodes[1].node.metadata)["date"] == "2020-01-03"
    # assert result_nodes[2].node.get_content() == "This is a test."
    # assert cast(Dict, result_nodes[2].node.metadata)["date"] == "2020-01-02"


def test_time_weighted_postprocessor() -> None:
    """Test time weighted processor."""
    key = "__last_accessed__"
    # try in metadata
    nodes = [
        TextNode(text="Hello world.", id_="1", metadata={key: 0}),
        TextNode(text="This is a test.", id_="2", metadata={key: 1}),
        TextNode(text="This is another test.", id_="3", metadata={key: 2}),
        TextNode(text="This is a test v2.", id_="4", metadata={key: 3}),
    ]
    node_with_scores = [NodeWithScore(node=node) for node in nodes]

    # high time decay
    postprocessor = TimeWeightedPostprocessor(
        top_k=1, time_decay=0.99999, time_access_refresh=True, now=4.0
    )
    result_nodes_with_score = postprocessor.postprocess_nodes(node_with_scores)

    assert len(result_nodes_with_score) == 1
    assert result_nodes_with_score[0].node.get_content() == "This is a test v2."
    assert cast(Dict, nodes[0].metadata)[key] == 0
    assert cast(Dict, nodes[3].metadata)[key] != 3

    # low time decay
    # artificially make earlier nodes more relevant
    # therefore postprocessor should still rank earlier nodes higher
    nodes = [
        TextNode(text="Hello world.", id_="1", metadata={key: 0}),
        TextNode(text="This is a test.", id_="2", metadata={key: 1}),
        TextNode(text="This is another test.", id_="3", metadata={key: 2}),
        TextNode(text="This is a test v2.", id_="4", metadata={key: 3}),
    ]
    node_with_scores = [
        NodeWithScore(node=node, score=-float(idx)) for idx, node in enumerate(nodes)
    ]
    postprocessor = TimeWeightedPostprocessor(
        top_k=1, time_decay=0.000000000002, time_access_refresh=True, now=4.0
    )
    result_nodes_with_score = postprocessor.postprocess_nodes(node_with_scores)
    assert len(result_nodes_with_score) == 1
    assert result_nodes_with_score[0].node.get_content() == "Hello world."
    assert cast(Dict, nodes[0].metadata)[key] != 0
    assert cast(Dict, nodes[3].metadata)[key] == 3


@pytest.mark.skipif(not spacy_installed, reason="spacy not installed")
def test_keyword_postprocessor() -> None:
    """Test keyword processor."""
    key = "__last_accessed__"
    # try in metadata
    nodes = [
        TextNode(text="Hello world.", id_="1", metadata={key: 0}),
        TextNode(text="This is a test.", id_="2", metadata={key: 1}),
        TextNode(text="This is another test.", id_="3", metadata={key: 2}),
        TextNode(text="This is a test v2.", id_="4", metadata={key: 3}),
    ]
    node_with_scores = [NodeWithScore(node=node) for node in nodes]

    postprocessor = KeywordNodePostprocessor(required_keywords=["This"])
    new_nodes = postprocessor.postprocess_nodes(node_with_scores)
    assert new_nodes[0].node.get_content() == "This is a test."
    assert new_nodes[1].node.get_content() == "This is another test."
    assert new_nodes[2].node.get_content() == "This is a test v2."

    postprocessor = KeywordNodePostprocessor(required_keywords=["Hello"])
    new_nodes = postprocessor.postprocess_nodes(node_with_scores)
    assert new_nodes[0].node.get_content() == "Hello world."
    assert len(new_nodes) == 1

    postprocessor = KeywordNodePostprocessor(required_keywords=["is another"])
    new_nodes = postprocessor.postprocess_nodes(node_with_scores)
    assert new_nodes[0].node.get_content() == "This is another test."
    assert len(new_nodes) == 1

    # test exclude keywords
    postprocessor = KeywordNodePostprocessor(exclude_keywords=["is another"])
    new_nodes = postprocessor.postprocess_nodes(node_with_scores)
    assert new_nodes[1].node.get_content() == "This is a test."
    assert new_nodes[2].node.get_content() == "This is a test v2."
    assert len(new_nodes) == 3


@pytest.mark.skipif(not spacy_installed, reason="spacy not installed")
def test_keyword_postprocessor_for_non_english() -> None:
    """Test keyword processor for non English."""
    try:
        key = "__last_accessed__"
        # try in metadata
        nodes = [
            TextNode(text="こんにちは世界。", id_="1", metadata={key: 0}),
            TextNode(text="これはテストです。", id_="2", metadata={key: 1}),
            TextNode(text="これは別のテストです。", id_="3", metadata={key: 2}),
            TextNode(text="これはテストv2です。", id_="4", metadata={key: 3}),
        ]
        node_with_scores = [NodeWithScore(node=node) for node in nodes]

        postprocessor = KeywordNodePostprocessor(required_keywords=["これ"], lang="ja")
        new_nodes = postprocessor.postprocess_nodes(node_with_scores)
        assert new_nodes[0].node.get_content() == "これはテストです。"
        assert new_nodes[1].node.get_content() == "これは別のテストです。"
        assert new_nodes[2].node.get_content() == "これはテストv2です。"

        postprocessor = KeywordNodePostprocessor(required_keywords=["別の"], lang="ja")
        new_nodes = postprocessor.postprocess_nodes(node_with_scores)
        assert new_nodes[0].node.get_content() == "これは別のテストです。"
        assert len(new_nodes) == 1

        # test exclude keywords
        postprocessor = KeywordNodePostprocessor(exclude_keywords=["別の"], lang="ja")
        new_nodes = postprocessor.postprocess_nodes(node_with_scores)
        assert new_nodes[1].node.get_content() == "これはテストです。"
        assert new_nodes[2].node.get_content() == "これはテストv2です。"
        assert len(new_nodes) == 3

        # test both required and exclude keywords
        postprocessor = KeywordNodePostprocessor(
            required_keywords=["テスト"], exclude_keywords=["v2"], lang="ja"
        )
        new_nodes = postprocessor.postprocess_nodes(node_with_scores)
        assert new_nodes[0].node.get_content() == "これはテストです。"
        assert new_nodes[1].node.get_content() == "これは別のテストです。"
        assert len(new_nodes) == 2
    except ImportError:
        pass
