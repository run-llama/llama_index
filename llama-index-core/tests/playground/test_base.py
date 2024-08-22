"""Test Playground."""

from typing import List

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.playground import (
    DEFAULT_INDEX_CLASSES,
    DEFAULT_MODES,
    Playground,
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


def test_get_set_compare(patch_llm_predictor, patch_token_text_splitter) -> None:
    """Test basic comparison of indices."""
    documents = [Document(text="They're taking the Hobbits to Isengard!")]

    indices = [
        VectorStoreIndex.from_documents(
            documents=documents, embed_model=MockEmbedding()
        ),
        SummaryIndex.from_documents(documents),
        TreeIndex.from_documents(documents=documents),
    ]

    playground = Playground(indices=indices)  # type: ignore

    assert len(playground.indices) == 3

    results = playground.compare("Who is?", to_pandas=False)
    assert len(results) > 0
    assert len(results) <= 3 * len(DEFAULT_MODES)

    playground.indices = [
        VectorStoreIndex.from_documents(
            documents=documents, embed_model=MockEmbedding()
        )
    ]

    assert len(playground.indices) == 1


def test_from_docs(patch_llm_predictor, patch_token_text_splitter) -> None:
    """Test initialization via a list of documents."""
    documents = [
        Document(text="I can't carry it for you."),
        Document(text="But I can carry you!"),
    ]

    playground = Playground.from_docs(documents=documents)

    assert len(playground.indices) == len(DEFAULT_INDEX_CLASSES)
    assert len(playground.retriever_modes) == len(DEFAULT_MODES)

    with pytest.raises(ValueError):
        playground = Playground.from_docs(documents=documents, retriever_modes={})


def test_validation() -> None:
    """Test validation of indices and modes."""
    with pytest.raises(ValueError):
        _ = Playground(indices=["VectorStoreIndex"])  # type: ignore

    with pytest.raises(ValueError):
        _ = Playground(
            indices=[VectorStoreIndex, SummaryIndex, TreeIndex]  # type: ignore
        )

    with pytest.raises(ValueError):
        _ = Playground(indices=[])  # type: ignore

    with pytest.raises(TypeError):
        _ = Playground(retriever_modes={})  # type: ignore
