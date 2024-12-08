import os
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.pinecone_native_rerank import PineconeNativeRerank


os.environ["PINECONE_API_KEY"] = "your-sdk"


txts = [
    "Apple is a popular fruit known for its sweetness and crisp texture.",
    "Apple is known for its innovative products like the iPhone.",
    "Many people enjoy eating apples as a healthy snack.",
    "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
    "An apple a day keeps the doctor away, as the saying goes.",
    "apple has a lot of vitamins",
]


nodes = [
    NodeWithScore(node=TextNode(id_=f"vec{i}", text=txt)) for i, txt in enumerate(txts)
]


query_bundle = QueryBundle(
    query_str="The tech company Apple is known for its innovative products like the iPhone."
)

reranker = PineconeNativeRerank(top_n=4, model="pinecone-rerank-v0")

result = reranker._postprocess_nodes(nodes=nodes, query_bundle=query_bundle)

for node_with_score in result:
    print(
        f"ID: {node_with_score.node.id_}, Score: {node_with_score.score}, Content: {node_with_score.node.get_content()}"
    )
