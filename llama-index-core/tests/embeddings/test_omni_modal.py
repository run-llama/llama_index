"""Embeddings."""
from dataclasses import dataclass, field
from typing import Any, List, Type

import pytest

from llama_index.core.base.embeddings.base import Embedding, SimilarityMode
from llama_index.core.embeddings.omni_modal_base import (
    K,
    KQ,
    KD,
    Modality,
    ModalityBundle,
    Modalities,
    NodeProcessor,
    NodeProcessors,
    OmniModalEmbedding,
    OmniModalEmbeddingBundle,
    QueryProcessor,
)
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


def _create_modality(name: K) -> Modality[K, MockNode, str]:
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
    def from_name(cls, name: K) -> Modality[K, MockNode, str]:
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
    bundle = ModalityBundle.of()

    assert not bundle
    assert len(bundle) == 0
    assert bundle == bundle

    assert bundle.keys() == set()


def test_modality_bundle_single():
    bundle = ModalityBundle.of(Modalities.TEXT)

    assert bundle
    assert len(bundle) == 1
    assert bundle == bundle

    assert bundle.keys() == {Modalities.TEXT.key}


def test_modality_bundle_two():
    bundle = ModalityBundle.of(Modalities.TEXT, Modalities.IMAGE)

    assert bundle
    assert len(bundle) == 2
    assert bundle == bundle

    assert bundle.keys() == {Modalities.TEXT.key, Modalities.IMAGE.key}

    bundle2 = ModalityBundle.of(Modalities.IMAGE, Modalities.TEXT)
    assert bundle == bundle2, "Key order should not matter"


def test_modality_bundle_duplicate():
    with pytest.raises(ValueError, match="duplicate modality keys"):
        ModalityBundle.of(Modalities.TEXT, Modalities.TEXT)

    with pytest.raises(ValueError, match="duplicate modality keys"):
        ModalityBundle.of(Modalities.TEXT, Modalities.IMAGE, Modalities.TEXT)


def _mock_default_document_modalities():
    raise ValueError("No document modalities were provided")


def _mock_default_query_modalities():
    raise ValueError("No query modalities were provided")


@dataclass
class MockTextEmbedding(OmniModalEmbedding[KD, KQ]):
    _document_modalities: ModalityBundle[KD] = field(
        default_factory=_mock_default_query_modalities
    )
    _query_modalities: ModalityBundle[KQ] = field(
        default_factory=_mock_default_document_modalities
    )

    @property
    def document_modalities(self) -> ModalityBundle[KD]:
        return self._document_modalities

    @property
    def query_modalities(self) -> ModalityBundle[KQ]:
        return self._query_modalities

    def _get_vector(self, text: str) -> Embedding:
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

    def _get_query_embedding(
        self, modality: Modality[KQ, Any, str], data: object
    ) -> Embedding:
        assert isinstance(data, str)
        return self._get_vector(data)

    def _get_document_embedding(
        self, modality: Modality[KD, Any, str], data: object
    ) -> Embedding:
        assert isinstance(data, str)
        return self._get_vector(data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(document_modalities={self.document_modalities}, query_modalities={self.query_modalities})"


@pytest.mark.parametrize("modality", [MockModalities.A])
def test_get_document_embeddings(modality: Modality[str, MockNode, str]) -> None:
    embed_model = MockTextEmbedding(
        _document_modalities=ModalityBundle.of(modality),
        _query_modalities=ModalityBundle.of(modality),
    )
    texts_to_embed = []
    for i in range(8):
        texts_to_embed.append("Hello world.")
    for i in range(8):
        texts_to_embed.append("This is a test.")
    for i in range(4):
        texts_to_embed.append("This is another test.")
    for i in range(4):
        texts_to_embed.append("This is a test v2.")

    doc_embeddings = [
        embed_model.get_document_embedding(modality.key, texts_to_embed[i])
        for i in range(24)
    ]
    for i in range(8):
        assert doc_embeddings[i] == [1, 0, 0, 0, 0]
    for i in range(8, 16):
        assert doc_embeddings[i] == [0, 1, 0, 0, 0]
    for i in range(16, 20):
        assert doc_embeddings[i] == [0, 0, 1, 0, 0]
    for i in range(20, 24):
        assert doc_embeddings[i] == [0, 0, 0, 1, 0]

    assert doc_embeddings == embed_model.get_document_embedding_batch(
        modality.key, texts_to_embed
    ), "Inconsistent with batch operation"


@pytest.mark.parametrize("modality", [MockModalities.A])
def test_get_query_embeddings(modality: Modality[str, MockNode, str]) -> None:
    embed_model = MockTextEmbedding(
        _document_modalities=ModalityBundle.of(modality),
        _query_modalities=ModalityBundle.of(modality),
    )
    texts_to_embed = []
    for i in range(8):
        texts_to_embed.append("Hello world.")
    for i in range(8):
        texts_to_embed.append("This is a test.")
    for i in range(4):
        texts_to_embed.append("This is another test.")
    for i in range(4):
        texts_to_embed.append("This is a test v2.")

    query_embeddings = [
        embed_model.get_query_embedding(modality.key, texts_to_embed[i])
        for i in range(24)
    ]
    for i in range(8):
        assert query_embeddings[i] == [1, 0, 0, 0, 0]
    for i in range(8, 16):
        assert query_embeddings[i] == [0, 1, 0, 0, 0]
    for i in range(16, 20):
        assert query_embeddings[i] == [0, 0, 1, 0, 0]
    for i in range(20, 24):
        assert query_embeddings[i] == [0, 0, 0, 1, 0]

    qe1 = embed_model.get_agg_embedding_from_queries(modality.key, texts_to_embed[0:8])
    qe2 = embed_model.get_agg_embedding_from_queries(modality.key, texts_to_embed[8:16])
    qe3 = embed_model.get_agg_embedding_from_queries(
        modality.key, texts_to_embed[16:20]
    )
    qe4 = embed_model.get_agg_embedding_from_queries(
        modality.key, texts_to_embed[20:24]
    )

    assert (
        query_embeddings == [qe1] * 8 + [qe2] * 8 + [qe3] * 4 + [qe4] * 4
    ), "Inconsistent with batch operation"


def test_get_document_embeddings_missing() -> None:
    embed_model = MockTextEmbedding(
        _document_modalities=ModalityBundle.of(MockModalities.A),
        _query_modalities=ModalityBundle.of(MockModalities.A),
    )

    with pytest.raises(ValueError, match=r"document modality .* is not supported"):
        embed_model.get_document_embedding(MockModalities.B.key, "")


def test_get_query_embeddings_missing() -> None:
    embed_model = MockTextEmbedding(
        _document_modalities=ModalityBundle.of(MockModalities.A),
        _query_modalities=ModalityBundle.of(MockModalities.A),
    )

    with pytest.raises(ValueError, match=r"query modality .* is not supported"):
        embed_model.get_query_embedding(MockModalities.B.key, "")


def test_embedding_similarity() -> None:
    embed_model = MockTextEmbedding(
        _document_modalities=ModalityBundle.of(),
        _query_modalities=ModalityBundle.of(),
    )
    text_embedding = [3.0, 4.0, 0.0]
    query_embedding = [0.0, 1.0, 0.0]
    cosine = embed_model.similarity(query_embedding, text_embedding)
    assert cosine == 0.8


def test_embedding_similarity_euclidean() -> None:
    embed_model = MockTextEmbedding(
        _document_modalities=ModalityBundle.of(),
        _query_modalities=ModalityBundle.of(),
    )
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


def test_embedding_bundle_empty():
    bundle = OmniModalEmbeddingBundle.of()

    assert not bundle
    assert len(bundle) == 0
    assert bundle == bundle

    assert bundle.document_modalities == ModalityBundle.of()
    assert bundle.query_modalities == ModalityBundle.of()


def test_embedding_bundle_single():
    bundle = OmniModalEmbeddingBundle.of(
        MockTextEmbedding(
            _document_modalities=ModalityBundle.of(MockModalities.A),
            _query_modalities=ModalityBundle.of(MockModalities.A),
        ),
    )

    assert bundle
    assert len(bundle) == 1
    assert bundle == bundle

    assert bundle.document_modalities == ModalityBundle.of(MockModalities.A)
    assert bundle.query_modalities == ModalityBundle.of(MockModalities.A)


def test_embedding_bundle_two():
    emb_a = MockTextEmbedding(
        _document_modalities=ModalityBundle.of(MockModalities.A),
        _query_modalities=ModalityBundle.of(MockModalities.A),
    )
    emb_b = MockTextEmbedding(
        _document_modalities=ModalityBundle.of(MockModalities.B),
        _query_modalities=ModalityBundle.of(MockModalities.B),
    )

    bundle = OmniModalEmbeddingBundle.of(emb_a, emb_b)

    assert bundle
    assert len(bundle) == 2
    assert bundle == bundle

    assert bundle.document_modalities == ModalityBundle.of(
        MockModalities.A, MockModalities.B
    )
    assert bundle.query_modalities == ModalityBundle.of(
        MockModalities.A, MockModalities.B
    )

    bundle2 = OmniModalEmbeddingBundle.of(emb_b, emb_a)
    assert bundle == bundle2, "Key order should not matter"


def test_embedding_bundle_duplicate():
    with pytest.raises(ValueError, match="duplicate document modalities"):
        OmniModalEmbeddingBundle.of(
            MockTextEmbedding(
                _document_modalities=ModalityBundle.of(MockModalities.A),
                _query_modalities=ModalityBundle.of(MockModalities.A),
            ),
            MockTextEmbedding(
                _document_modalities=ModalityBundle.of(MockModalities.A),
                _query_modalities=ModalityBundle.of(MockModalities.B),
            ),
        )

    with pytest.raises(ValueError, match="duplicate document modalities"):
        OmniModalEmbeddingBundle.of(
            MockTextEmbedding(
                _document_modalities=ModalityBundle.of(MockModalities.A),
                _query_modalities=ModalityBundle.of(MockModalities.A),
            ),
            MockTextEmbedding(
                _document_modalities=ModalityBundle.of(MockModalities.B),
                _query_modalities=ModalityBundle.of(MockModalities.B),
            ),
            MockTextEmbedding(
                _document_modalities=ModalityBundle.of(MockModalities.A),
                _query_modalities=ModalityBundle.of(MockModalities.B),
            ),
        )

    bundle = OmniModalEmbeddingBundle.of(
        MockTextEmbedding(
            _document_modalities=ModalityBundle.of(MockModalities.A),
            _query_modalities=ModalityBundle.of(MockModalities.A),
        ),
        MockTextEmbedding(
            _document_modalities=ModalityBundle.of(MockModalities.B),
            _query_modalities=ModalityBundle.of(MockModalities.A),
        ),
    )

    assert bundle.document_modalities == ModalityBundle.of(
        MockModalities.A, MockModalities.B
    )
    assert bundle.query_modalities == ModalityBundle.of(MockModalities.A)
