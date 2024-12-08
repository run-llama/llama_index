import os
from unittest.mock import patch
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.postprocessor.pinecone_native_rerank import PineconeNativeRerank

os.environ["PINECONE_API_KEY"] = "your-sdk"


def test_pinecone_native_reranker():
    names_of_base_classes = [b.__name__ for b in PineconeNativeRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_pinecone_native_reranker_initialization():
    reranker = PineconeNativeRerank(top_n=4, model="pinecone-rerank-v0")

    assert reranker.top_n == 4
    assert reranker.model == "pinecone-rerank-v0"


@patch.dict(os.environ, {"PINECONE_API_KEY": "mocked-key"})
@patch(
    "llama_index.postprocessor.pinecone_native_rerank.PineconeNativeRerank._postprocess_nodes"
)
def test_pinecone_native_reranker_postprocess_nodes(mock_postprocess_nodes):
    mock_postprocess_nodes.return_value = [
        NodeWithScore(node=TextNode(id_="vec0", text="Mocked text 1"), score=0.9),
        NodeWithScore(node=TextNode(id_="vec1", text="Mocked text 2"), score=0.8),
        NodeWithScore(node=TextNode(id_="vec2", text="Mocked text 3"), score=0.7),
        NodeWithScore(node=TextNode(id_="vec3", text="Mocked text 4"), score=0.6),
    ]

    txts = [
        "Apple is a popular fruit known for its sweetness and crisp texture.",
        "Apple is known for its innovative products like the iPhone.",
        "Many people enjoy eating apples as a healthy snack.",
        "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
        "An apple a day keeps the doctor away, as the saying goes.",
        "apple has a lot of vitamins",
    ]
    nodes = [
        NodeWithScore(node=TextNode(id_=f"vec{i}", text=txt))
        for i, txt in enumerate(txts)
    ]
    query_bundle = QueryBundle(
        query_str="The tech company Apple is known for its innovative products like the iPhone."
    )
    reranker = PineconeNativeRerank(top_n=4, model="pinecone-rerank-v0")
    result = reranker._postprocess_nodes(nodes=nodes, query_bundle=query_bundle)

    assert len(result) == 4
    for node_with_score in result:
        assert isinstance(node_with_score.node, TextNode)
        assert isinstance(node_with_score.score, float)
