from llama_index.core.node_parser.text.sentence_window import (
    SentenceWindowNodeParser,
)
from llama_index.core.schema import Document


def test_split_and_window() -> None:
    document = Document(text="This is a test 1. This is a test 2. This is a test 3.")

    node_parser = SentenceWindowNodeParser.from_defaults()

    nodes = node_parser.get_nodes_from_documents([document])

    assert len(nodes) == 3
    assert nodes[0].get_content() == "This is a test 1. "
    assert nodes[1].get_content() == "This is a test 2. "
    assert nodes[2].get_content() == "This is a test 3."

    assert (
        nodes[0].metadata["window"]
        == "This is a test 1.  This is a test 2.  This is a test 3."
    )
    assert nodes[0].metadata["original_text"] == "This is a test 1. "


def test_sentence_window_no_duplicate_excluded_metadata_keys() -> None:
    document = Document(
        text="This is a test 1. This is a test 2.",
        excluded_embed_metadata_keys=["custom_embed", "window"],
        excluded_llm_metadata_keys=["custom_llm", "original_text"],
    )
    node_parser = SentenceWindowNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents([document])

    assert len(nodes) == 2
    for node in nodes:
        assert node.excluded_embed_metadata_keys == [
            "custom_embed",
            "window",
            "original_text",
        ]
        assert node.excluded_llm_metadata_keys == [
            "custom_llm",
            "original_text",
            "window",
        ]
        assert len(node.excluded_embed_metadata_keys) == len(
            set(node.excluded_embed_metadata_keys)
        )
        assert len(node.excluded_llm_metadata_keys) == len(
            set(node.excluded_llm_metadata_keys)
        )
