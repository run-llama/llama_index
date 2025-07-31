from llama_index.core import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.retrievers.bm25.base import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter


def test_class():
    names_of_base_classes = [b.__name__ for b in BM25Retriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes


def test_scores():
    documents = [
        Document(text="Large Language Model"),
        Document(text="LlamaIndex is a data framework for your LLM application"),
        Document(text="How to use LlamaIndex"),
    ]

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
    result_nodes = retriever.retrieve("llamaindex llm")
    assert len(result_nodes) == 2
    for node in result_nodes:
        assert node.score is not None
        assert node.score > 0.0


def test_large_value_of_top_k():
    documents = [Document.example()]

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # Passing a high value of similarity_top_k w.r.t Document example
    similarity_top_k = 20
    retriever = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=similarity_top_k
    )
    result_nodes = retriever.retrieve("What is llama index about?")

    # As we had less nodes then similarity_top_k in the retriever
    assert len(result_nodes) < similarity_top_k
    # Retrieved nodes should be all the nodes added in retriever
    assert len(result_nodes) == len(nodes)
