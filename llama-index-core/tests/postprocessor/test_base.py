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
