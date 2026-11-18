from llama_index.core import Document, MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.node_parser.topic import TopicNodeParser


class StubTopicNodeParser(TopicNodeParser):
    """TopicNodeParser with the LLM/embedding round-trips stubbed out."""

    def proposition_transfer(self, paragraph):
        # One proposition per paragraph keeps the chunk arithmetic exact.
        return [paragraph]

    def is_same_topic_llm(self, current_chunk, new_proposition):
        return True

    def is_same_topic_embedding(self, current_chunk, new_proposition):
        return True


def _word_tokenizer(text):
    return text.split()


def test_llm_chunking():
    llm = MockLLM()
    embed_model = MockEmbedding(embed_dim=3)
    node_parser = TopicNodeParser.from_defaults(
        llm=llm, embed_model=embed_model, similarity_method="llm"
    )

    nodes = node_parser([Document(text="Hello world!"), Document(text="Hello world!")])
    print(nodes)
    # MockLLM echoes the proposition example from the system prompt; it fits in one
    # chunk per document under the documented token budget.
    assert len(nodes) == 2


def test_chunk_size_is_measured_in_tokens():
    """`max_chunk_size` is a token budget, so chunks may be filled up to that budget."""
    words = [
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
    ]
    paragraphs = []
    for i in range(20):
        paragraph_words = words * 10
        paragraph_words[0] = f"p{i}"
        paragraph = " ".join(paragraph_words)
        assert len(_word_tokenizer(paragraph)) == 100
        paragraphs.append(paragraph)

    node_parser = StubTopicNodeParser.from_defaults(
        llm=MockLLM(),
        embed_model=MockEmbedding(embed_dim=3),
        tokenizer=_word_tokenizer,
        max_chunk_size=1000,
        window_size=1,
    )

    chunks = node_parser.semantic_chunking(paragraphs)

    # 20 x 100 tokens fit into exactly two 1000-token chunks.
    assert [len(_word_tokenizer(chunk)) for chunk in chunks] == [1000, 1000]


def test_embedding_chunking():
    llm = MockLLM()
    embed_model = MockEmbedding(embed_dim=3)
    node_parser = TopicNodeParser.from_defaults(
        llm=llm, embed_model=embed_model, similarity_method="embedding"
    )

    nodes = node_parser([Document(text="Hello world!"), Document(text="Hello world!")])
    print(nodes)
    # MockLLM echoes the proposition example from the system prompt; it fits in one
    # chunk per document under the documented token budget.
    assert len(nodes) == 2
