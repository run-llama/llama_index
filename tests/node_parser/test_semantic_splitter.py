from llama_index.node_parser.text.semantic_splitter import SemanticSplitterNodeParser
from llama_index.schema import Document
from tests.playground.test_base import MockEmbedding


def test_grouped_semantically() -> None:
    document = Document(
        text="They're taking the Hobbits to Isengard! I can't carry it for you. But I can carry you!"
    )

    embeddings = MockEmbedding()

    node_parser = SemanticSplitterNodeParser.from_defaults(embeddings)

    nodes = node_parser.get_nodes_from_documents([document])

    assert len(nodes) == 1
    assert (
        nodes[0].get_content()
        == "They're taking the Hobbits to Isengard! I can't carry it for you. But I can carry you!"
    )


def test_split_and_permutated() -> None:
    document = Document(
        text="They're taking the Hobbits to Isengard! I can't carry it for you. But I can carry you!"
    )

    embeddings = MockEmbedding()

    node_parser = SemanticSplitterNodeParser.from_defaults(embeddings)

    text_splits = node_parser.sentence_splitter(document.text)

    sentences = node_parser._build_sentence_groups(text_splits)

    assert len(sentences) == 3
    assert sentences[0]["sentence"] == "They're taking the Hobbits to Isengard! "
    assert (
        sentences[0]["combined_sentence"]
        == "They're taking the Hobbits to Isengard! I can't carry it for you. "
    )
    assert sentences[1]["sentence"] == "I can't carry it for you. "
    assert (
        sentences[1]["combined_sentence"]
        == "They're taking the Hobbits to Isengard! I can't carry it for you. But I can carry you!"
    )
    assert sentences[2]["sentence"] == "But I can carry you!"
    assert (
        sentences[2]["combined_sentence"]
        == "I can't carry it for you. But I can carry you!"
    )
