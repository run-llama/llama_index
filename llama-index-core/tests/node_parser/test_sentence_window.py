from typing import List

import pytest
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


@pytest.mark.parametrize(
    "sentences",
    [
        ["भारत एक विशाल देश है। ", "इसकी राजधानी नई दिल्ली है। ", "गंगा एक पवित्र नदी है।"],
        ["मुंबई ही राजधानी आहे। ", "पुणे हे शिक्षणाचे माहेरघर आहे। ", "ते धान्य पिकवतात॥"],
        ["বাংলা একটি সুন্দর ভাষা। ", "কলকাতা পশ্চিমবঙ্গের রাজধানী। ", "এখানে অনেক মানুষ থাকে।"],
    ],
    ids=["hindi", "marathi", "bengali"],
)
def test_split_and_window_on_danda(sentences: List[str]) -> None:
    document = Document(text="".join(sentences))

    nodes = SentenceWindowNodeParser.from_defaults().get_nodes_from_documents(
        [document]
    )

    assert [node.get_content() for node in nodes] == sentences
