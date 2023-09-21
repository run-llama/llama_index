from llama_index.schema import TextNode, NodeWithScore
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor


def test_metadata_replacement() -> None:
    node = TextNode(
        text="This is a test 1.", metadata={"key": "This is a another test."}
    )

    nodes = [NodeWithScore(node=node, score=1.0)]

    postprocessor = MetadataReplacementPostProcessor(target_metadata_key="key")

    nodes = postprocessor.postprocess_nodes(nodes)

    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is a another test."
