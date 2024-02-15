from typing import List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser.text.semantic_splitter import (
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import Document


class MockEmbedding(BaseEmbedding):
    @classmethod
    def class_name(cls) -> str:
        return "MockEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        del query
        return [0, 0, 1, 0, 0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        text = text.strip()
        # assume dimensions are 5
        if text == "They're taking the Hobbits to Isengard!":
            return [1, 0, 0, 0, 0]
        elif (
            text == "They're taking the Hobbits to Isengard! I can't carry it for you."
        ):
            return [1, 1, 0, 0, 0]
        elif (
            text
            == "They're taking the Hobbits to Isengard! I can't carry it for you. But I can carry you!"
        ):
            return [1, 1, 1, 0, 0]
        elif text == "I can't carry it for you.":
            return [0, 1, 0, 0, 0]
        elif text == "I can't carry it for you. But I can carry you!":
            return [0, 1, 1, 0, 0]
        elif text == "But I can carry you!":
            return [0, 0, 1, 0, 0]
        else:
            print(text)
            raise ValueError(f"Invalid text for `mock_get_text_embedding`.")

    def _get_text_embedding(self, text: str) -> List[float]:
        """Mock get text embedding."""
        text = text.strip()
        # assume dimensions are 5
        if text == "They're taking the Hobbits to Isengard!":
            return [1, 0, 0, 0, 0]
        elif (
            text == "They're taking the Hobbits to Isengard! I can't carry it for you."
        ):
            return [1, 1, 0, 0, 0]
        elif (
            text
            == "They're taking the Hobbits to Isengard! I can't carry it for you. But I can carry you!"
        ):
            return [1, 1, 1, 0, 0]
        elif text == "I can't carry it for you.":
            return [0, 1, 0, 0, 0]
        elif text == "I can't carry it for you. But I can carry you!":
            return [0, 1, 1, 0, 0]
        elif text == "But I can carry you!":
            return [0, 0, 1, 0, 0]
        else:
            print(text)
            raise ValueError("Invalid text for `mock_get_text_embedding`.")

    def _get_query_embedding(self, query: str) -> List[float]:
        """Mock get query embedding."""
        del query
        return [0, 0, 1, 0, 0]


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
