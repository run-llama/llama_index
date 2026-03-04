from llama_index.core.schema import (
    MediaResource,
    Node,
    NodeWithScore,
    QueryBundle,
)
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank


def test_init():
    reranker = FlashRankRerank()
    assert reranker is not None


def test_postprocess_nodes():
    reranker = FlashRankRerank()

    query_bundle = QueryBundle(
        query_str="I'm visiting New York City, what is the best place to get Bagels?"
    )

    # Simulate bad results from a poor embedding model / query
    node_one = NodeWithScore(
        node=Node(
            text_resource=MediaResource(text="I'm just a Node i am only a Node"),
            id_="1",
        ),
        score=0.9,
    )
    node_two = NodeWithScore(
        node=Node(
            text_resource=MediaResource(
                text="The best place to get Bagels in New York City"
            ),
            id_="2",
        ),
        score=0.3,
    )
    node_three = NodeWithScore(
        node=Node(
            text_resource=MediaResource(
                text="A Latte without milk is just an espresso"
            ),
            id_="3",
        ),
        score=1.0,
    )

    nodes = [node_one, node_two, node_three]

    reranked_nodes = reranker.postprocess_nodes(nodes, query_bundle)

    assert reranked_nodes is not None
    assert len(reranked_nodes) == 3
    assert reranked_nodes[0] == node_two
    assert reranked_nodes[1] == node_one
    assert reranked_nodes[2] == node_three
