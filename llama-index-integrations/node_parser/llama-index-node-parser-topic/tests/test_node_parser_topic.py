from llama_index.core import Document, MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.node_parser.topic import TopicNodeParser


def test_llm_chunking():
    llm = MockLLM()
    embed_model = MockEmbedding(embed_dim=3)
    node_parser = TopicNodeParser.from_defaults(
        llm=llm, embed_model=embed_model, similarity_method="llm"
    )

    nodes = node_parser([Document(text="Hello world!"), Document(text="Hello world!")])
    print(nodes)
    assert len(nodes) == 4


def test_embedding_chunking():
    llm = MockLLM()
    embed_model = MockEmbedding(embed_dim=3)
    node_parser = TopicNodeParser.from_defaults(
        llm=llm, embed_model=embed_model, similarity_method="embedding"
    )

    nodes = node_parser([Document(text="Hello world!"), Document(text="Hello world!")])
    print(nodes)
    assert len(nodes) == 4
