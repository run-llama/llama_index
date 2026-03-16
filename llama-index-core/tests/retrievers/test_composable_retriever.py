from llama_index.core.indices import SummaryIndex
from llama_index.core.schema import IndexNode, TextNode


def test_composable_retrieval() -> None:
    """Test composable retrieval."""
    text_node = TextNode(text="This is a test text node.", id_="test_text_node")
    index_node = IndexNode(
        text="This is a test index node.",
        id_="test_index_node",
        index_id="test_index_node_index",
        obj=TextNode(text="Hidden node!", id_="hidden_node"),
    )

    index = SummaryIndex(nodes=[text_node, text_node], objects=[index_node])

    # Test retrieval
    retriever = index.as_retriever()
    nodes = retriever.retrieve("test")

    assert len(nodes) == 2
    assert nodes[0].node.id_ == "test_text_node"
    assert nodes[1].node.id_ == "hidden_node"
