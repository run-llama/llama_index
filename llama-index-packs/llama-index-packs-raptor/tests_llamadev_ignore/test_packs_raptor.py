from llama_index.core import Document, MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.packs.raptor.base import RaptorRetriever


def test_raptor() -> None:
    retriever = RaptorRetriever(
        [
            Document(text="one"),
            Document(text="two"),
            Document(text="three"),
            Document(text="four"),
            Document(text="five"),
            Document(text="six"),
            Document(text="seven"),
            Document(text="eight"),
            Document(text="nine"),
            Document(text="ten"),
        ],
        embed_model=MockEmbedding(embed_dim=1536),
        llm=MockLLM(),
    )

    assert len(retriever.index.docstore.docs) == 13

    nodes = retriever.retrieve("test", mode="collapsed")
    assert len(nodes) == 2

    nodes = retriever.retrieve("text", mode="tree_traversal")
    assert len(nodes) == 5
