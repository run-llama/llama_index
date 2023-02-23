"""Test Playground."""

from typing import Any, List
from unittest.mock import patch

import pytest

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.indices.vector_store import GPTSimpleVectorIndex
from gpt_index.playground import DEFAULT_INDEX_CLASSES, DEFAULT_MODES, Playground
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "They're taking the Hobbits to Isengard!":
        return [1, 0, 0, 0, 0]
    elif text == "I can't carry it for you.":
        return [0, 1, 0, 0, 0]
    elif text == "But I can carry you!":
        return [0, 0, 1, 0, 0]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_query_embedding(query: str) -> List[float]:
    """Mock get query embedding."""
    return [0, 0, 1, 0, 0]


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "_get_query_embedding", side_effect=mock_get_query_embedding
)
def test_get_set_compare(
    _mock_query_embed: Any,
    _mock_texts_embed: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test basic comparison of indices."""
    documents = [Document("They're taking the Hobbits to Isengard!")]

    indices = [
        GPTSimpleVectorIndex(documents=documents),
        GPTListIndex(documents=documents),
        GPTTreeIndex(documents=documents),
    ]

    playground = Playground(indices=indices)  # type: ignore

    assert len(playground.indices) == 3
    assert len(playground.modes) == len(DEFAULT_MODES)

    results = playground.compare("Who is?", to_pandas=False)
    assert len(results) > 0
    assert len(results) <= 3 * len(DEFAULT_MODES)

    playground.indices = [GPTSimpleVectorIndex(documents=documents)]
    playground.modes = ["default", "summarize"]

    assert len(playground.indices) == 1
    assert len(playground.modes) == 2

    with pytest.raises(ValueError):
        playground.modes = []


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_from_docs(
    _mock_embeds: Any,
    _mock_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
) -> None:
    """Test initialization via a list of documents."""
    documents = [
        Document("I can't carry it for you."),
        Document("But I can carry you!"),
    ]

    playground = Playground.from_docs(documents=documents)

    assert len(playground.indices) == len(DEFAULT_INDEX_CLASSES)
    assert len(playground.modes) == len(DEFAULT_MODES)

    with pytest.raises(ValueError):
        playground = Playground.from_docs(documents=documents, modes=[])


def test_validation() -> None:
    """Test validation of indices and modes."""
    with pytest.raises(ValueError):
        _ = Playground(indices=["GPTSimpleVectorIndex"])  # type: ignore

    with pytest.raises(ValueError):
        _ = Playground(
            indices=[GPTSimpleVectorIndex, GPTListIndex, GPTTreeIndex]  # type: ignore
        )

    with pytest.raises(ValueError):
        _ = Playground(indices=[])  # type: ignore

    with pytest.raises(TypeError):
        _ = Playground(modes=["default"])  # type: ignore
