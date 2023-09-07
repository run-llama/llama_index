
import pytest
from llama_index.schema import TextNode, NodeWithScore

@pytest.fixture
def text_node() -> TextNode:
    return TextNode(
        text='hello world',
        metadata={'foo': 'bar'},
        embedding=[0.1, 0.2, 0.3],
    )

@pytest.fixture
def node_with_score(text_node: TextNode) -> NodeWithScore:
    return NodeWithScore(
        node=text_node,
        score=0.5,
    )

def test_node_with_score_passthrough(node_with_score: NodeWithScore) -> None:
    _ = node_with_score.id_
    _ = node_with_score.node_id
    _ = node_with_score.text
    _ = node_with_score.metadata
    _ = node_with_score.embedding
    _ = node_with_score.get_text()
    _ = node_with_score.get_content()
    _ = node_with_score.get_embedding()



