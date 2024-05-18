"""Embeddings."""
from typing import Any, List, Type
from unittest.mock import patch

import pytest

from llama_index.core.base.embeddings.base import SimilarityMode
from llama_index.core.embeddings.omni_modal_base import (
    Modality,
    ModalityBundle,
    Modalities,
    NodeProcessor,
    NodeProcessors,
    OmniModalEmbedding,
    QueryProcessor,
)
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.utils import full_groupby


class MockNode(TextNode):
    def __repr__(self) -> str:
        return f"{type(self).__name__}(id_={self.id_})"


class MockNodeTypes:
    class A(MockNode):
        pass

    class AA(A):
        pass

    class B(MockNode):
        pass

    class C(MockNode):
        pass

    class BC(B, C):
        pass

    class CAB(C, A, B):
        pass

    @classmethod
    def from_name(cls, name: str) -> Type[MockNode]:
        return getattr(cls, name)

    @classmethod
    def create_nodes(cls) -> List[MockNode]:
        return [
            MockNodeTypes.A(),
            MockNodeTypes.C(),
            MockNodeTypes.B(),
            MockNodeTypes.AA(),
            MockNodeTypes.B(),
            MockNodeTypes.BC(),
            MockNodeTypes.BC(),
            MockNodeTypes.B(),
            MockNodeTypes.CAB(),
            MockNodeTypes.A(),
            MockNodeTypes.C(),
            MockNodeTypes.C(),
        ]


def _create_node_processor(name: str) -> NodeProcessor[MockNode, str]:
    return NodeProcessor(MockNodeTypes.from_name(name), lambda node: node.get_content())


class MockNodeProcessors:
    A = _create_node_processor("A")
    AA = _create_node_processor("AA")
    B = _create_node_processor("B")
    C = _create_node_processor("C")
    BC = _create_node_processor("BC")
    CAB = _create_node_processor("CAB")

    @classmethod
    def all(cls) -> List[NodeProcessor[MockNode, str]]:
        return [
            cls.A,
            cls.AA,
            cls.B,
            cls.C,
            cls.BC,
            cls.CAB,
        ]

    @classmethod
    def bases(cls) -> List[NodeProcessor[MockNode, str]]:
        return [
            cls.A,
            cls.B,
            cls.C,
        ]

    @classmethod
    def from_name(cls, name: str) -> NodeProcessor[MockNode, str]:
        return getattr(cls, name)


def _create_modality(name: str) -> Modality[str, MockNode, str]:
    return Modality(
        name,
        MockNodeProcessors.from_name(name),
        query_processor=QueryProcessor(lambda _: []),
    )


class MockModalities:
    A = _create_modality("A")
    AA = _create_modality("AA")
    B = _create_modality("B")
    C = _create_modality("C")
    BC = _create_modality("BC")
    CAB = _create_modality("CAB")

    @classmethod
    def all(cls) -> List[Modality[str, MockNode, str]]:
        return [
            cls.A,
            cls.AA,
            cls.B,
            cls.C,
            cls.BC,
            cls.CAB,
        ]

    @classmethod
    def bases(cls) -> List[Modality[str, MockNode, str]]:
        return [
            cls.A,
            cls.B,
            cls.C,
        ]

    @classmethod
    def from_name(cls, name: str) -> Modality[str, MockNode, str]:
        return getattr(cls, name)


def test_group_nodes_by_processor_exact_match():
    nodes = MockNodeTypes.create_nodes()

    actual = dict(NodeProcessors.group_nodes(nodes, MockNodeProcessors.all()))
    expected = {
        MockNodeProcessors.from_name(name): list(ns)
        for name, ns in full_groupby(nodes, key=lambda node: type(node).__name__)
    }

    assert actual == expected


def test_group_nodes_by_processor_partial_match():
    nodes = MockNodeTypes.create_nodes()

    actual = dict(NodeProcessors.group_nodes(nodes, MockNodeProcessors.bases()))
    expected = {
        MockNodeProcessors.from_name(name): list(ns)
        for name, ns in full_groupby(nodes, key=lambda node: type(node).__name__[0])
    }

    assert actual == expected


def test_group_nodes_by_modality_exact_match():
    nodes = MockNodeTypes.create_nodes()

    actual = dict(Modalities.group_nodes(nodes, MockModalities.all()))
    expected = {
        MockModalities.from_name(name): list(ns)
        for name, ns in full_groupby(nodes, key=lambda node: type(node).__name__)
    }

    assert actual == expected


def test_group_nodes_by_modality_partial_match():
    nodes = MockNodeTypes.create_nodes()

    actual = dict(Modalities.group_nodes(nodes, MockModalities.bases()))
    expected = {
        MockModalities.from_name(name): list(ns)
        for name, ns in full_groupby(nodes, key=lambda node: type(node).__name__[0])
    }

    assert actual == expected


def test_modality_bundle_empty():
    bundle = ModalityBundle()
    assert not bundle
    assert len(bundle) == 0
    assert bundle == bundle


def test_modality_bundle_single():
    bundle = ModalityBundle(Modalities.TEXT)
    assert bundle
    assert len(bundle) == 1
    assert bundle == bundle


def test_modality_bundle_two():
    bundle = ModalityBundle(Modalities.TEXT, Modalities.IMAGE)
    assert bundle
    assert len(bundle) == 2
    assert bundle == ModalityBundle(Modalities.IMAGE, Modalities.TEXT)


def test_modality_bundle_duplicate():
    with pytest.raises(ValueError, match="duplicate modality keys"):
        ModalityBundle(Modalities.TEXT, Modalities.TEXT)

    with pytest.raises(ValueError, match="duplicate modality keys"):
        ModalityBundle(Modalities.TEXT, Modalities.IMAGE, Modalities.TEXT)


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "Hello world.":
        return [1, 0, 0, 0, 0]
    elif text == "This is a test.":
        return [0, 1, 0, 0, 0]
    elif text == "This is another test.":
        return [0, 0, 1, 0, 0]
    elif text == "This is a test v2.":
        return [0, 0, 0, 1, 0]
    elif text == "This is a test v3.":
        return [0, 0, 0, 0, 1]
    elif text == "This is bar test.":
        return [0, 0, 1, 0, 0]
    elif text == "Hello world backup.":
        # this is used when "Hello world." is deleted.
        return [1, 0, 0, 0, 0]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


@patch.object(MockEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding)
@patch.object(
    MockEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_get_text_embeddings(
    _mock_get_text_embeddings: Any, _mock_get_text_embedding: Any
) -> None:
    """Test get queued text embeddings."""
    embed_model = OmniModalEmbedding.from_base(MockEmbedding(embed_dim=8))
    texts_to_embed = []
    for i in range(8):
        texts_to_embed.append("Hello world.")
    for i in range(8):
        texts_to_embed.append("This is a test.")
    for i in range(4):
        texts_to_embed.append("This is another test.")
    for i in range(4):
        texts_to_embed.append("This is a test v2.")

    result_embeddings = embed_model.get_document_embedding_batch(
        Modalities.TEXT.key, texts_to_embed
    )
    for i in range(8):
        assert result_embeddings[i] == [1, 0, 0, 0, 0]
    for i in range(8, 16):
        assert result_embeddings[i] == [0, 1, 0, 0, 0]
    for i in range(16, 20):
        assert result_embeddings[i] == [0, 0, 1, 0, 0]
    for i in range(20, 24):
        assert result_embeddings[i] == [0, 0, 0, 1, 0]


def test_embedding_similarity() -> None:
    """Test embedding similarity."""
    embed_model = OmniModalEmbedding.from_base(MockEmbedding(embed_dim=3))
    text_embedding = [3.0, 4.0, 0.0]
    query_embedding = [0.0, 1.0, 0.0]
    cosine = embed_model.similarity(query_embedding, text_embedding)
    assert cosine == 0.8


def test_embedding_similarity_euclidean() -> None:
    embed_model = OmniModalEmbedding.from_base(MockEmbedding(embed_dim=2))
    query_embedding = [1.0, 0.0]
    text1_embedding = [0.0, 1.0]  # further from query_embedding distance=1.414
    text2_embedding = [1.0, 1.0]  # closer to query_embedding distance=1.0
    euclidean_similarity1 = embed_model.similarity(
        query_embedding, text1_embedding, mode=SimilarityMode.EUCLIDEAN
    )
    euclidean_similarity2 = embed_model.similarity(
        query_embedding, text2_embedding, mode=SimilarityMode.EUCLIDEAN
    )
    assert euclidean_similarity1 < euclidean_similarity2
