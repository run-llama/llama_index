from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.schema import NodeWithScore, TextNode


def test_metadata_replacement() -> None:
    node = TextNode(
        text="This is a test 1.", metadata={"key": "This is a another test."}
    )

    nodes = [NodeWithScore(node=node, score=1.0)]

    postprocessor = MetadataReplacementPostProcessor(target_metadata_key="key")

    nodes = postprocessor.postprocess_nodes(nodes)

    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is a another test."


def test_metadata_replacement_falls_back_when_target_value_is_none() -> None:
    """A key present with an explicit None means no replacement, so the content stands."""
    node = TextNode(text="This is a test 1.", metadata={"key": None})

    nodes = [NodeWithScore(node=node, score=1.0)]

    postprocessor = MetadataReplacementPostProcessor(target_metadata_key="key")

    nodes = postprocessor.postprocess_nodes(nodes)

    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is a test 1."


def test_metadata_replacement_falls_back_when_target_key_is_absent() -> None:
    node = TextNode(text="This is a test 1.", metadata={"other": "unrelated"})

    nodes = [NodeWithScore(node=node, score=1.0)]

    postprocessor = MetadataReplacementPostProcessor(target_metadata_key="key")

    nodes = postprocessor.postprocess_nodes(nodes)

    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is a test 1."


def test_metadata_replacement_does_not_rewrite_cached_retrieval_result() -> None:
    original = NodeWithScore(
        node=TextNode(text="matched sentence", metadata={"window": "window text"}),
        score=0.75,
    )
    postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")

    [expanded] = postprocessor.postprocess_nodes([original])

    assert expanded.get_content() == "window text"
    assert expanded.score == original.score
    assert expanded.node_id == original.node_id
    assert original.get_content() == "matched sentence"
