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
