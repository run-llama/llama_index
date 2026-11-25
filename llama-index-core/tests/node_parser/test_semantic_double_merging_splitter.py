import zlib
from typing import Any, List

import pytest

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.node_parser.text.semantic_double_merging_splitter import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
from llama_index.core.schema import Document

doc = Document(
    text="Warsaw: Warsaw, the capital city of Poland, is a bustling metropolis located on the banks of the Vistula River. "
    "It is known for its rich history, vibrant culture, and resilient spirit. Warsaw's skyline is characterized by a mix of historic architecture and modern skyscrapers. "
    "The Old Town, with its cobblestone streets and colorful buildings, is a UNESCO World Heritage Site.\n\n"
    "Football: Football, also known as soccer, is a popular sport played by millions of people worldwide. "
    "It is a team sport that involves two teams of eleven players each. The objective of the game is to score goals by kicking the ball into the opposing team's goal. "
    "Football matches are typically played on a rectangular field called a pitch, with goals at each end. "
    "The game is governed by a set of rules known as the Laws of the Game. Football is known for its passionate fanbase and intense rivalries between clubs and countries. "
    "The FIFA World Cup is the most prestigious international football tournament.\n\n"
    "Mathematics: Mathematics is a fundamental discipline that deals with the study of numbers, quantities, and shapes. "
    "Its branches include algebra, calculus, geometry, and statistics."
)

doc_same = Document(
    text="Krakow is one of the oldest and largest cities in Poland, located in the southern part of the country on the Vistula River. "
    * 20
)

try:
    splitter = SemanticDoubleMergingSplitterNodeParser(
        initial_threshold=0.7,
        appending_threshold=0.8,
        merging_threshold=0.7,
        max_chunk_size=1000,
    )
    splitter.language_config.load_model()
    spacy_available = True
except Exception:
    spacy_available = False


@pytest.mark.skipif(not spacy_available, reason="Spacy model not available")
def test_number_of_returned_nodes() -> None:
    nodes = splitter.get_nodes_from_documents([doc])

    assert len(nodes) == 2


@pytest.mark.skipif(not spacy_available, reason="Spacy model not available")
def test_creating_initial_chunks() -> None:
    text = doc.text
    sentences = splitter.sentence_splitter(text)
    initial_chunks = splitter._create_initial_chunks(sentences)

    assert len(initial_chunks) == 4


@pytest.mark.skipif(not spacy_available, reason="Spacy model not available")
def test_config_models() -> None:
    with pytest.raises(ValueError):
        LanguageConfig(language="polish")

    with pytest.raises(ValueError):
        LanguageConfig(language="polish", spacy_model="en_core_web_md")

    with pytest.raises(ValueError):
        LanguageConfig(language="french", spacy_model="en_core_web_md")

    with pytest.raises(ValueError):
        LanguageConfig(language="empty", spacy_model="empty")

    LanguageConfig(language="english", spacy_model="en_core_web_md")


@pytest.mark.skipif(not spacy_available, reason="Spacy model not available")
def test_chunk_size_1() -> None:
    splitter.max_chunk_size = 0
    nodes = splitter.get_nodes_from_documents([doc])

    # length of each sentence
    assert len(nodes) == 13
    assert len(nodes[0].get_content()) == 111
    assert len(nodes[1].get_content()) == 72
    assert len(nodes[2].get_content()) == 91
    assert len(nodes[3].get_content()) == 99
    assert len(nodes[4].get_content()) == 100
    assert len(nodes[5].get_content()) == 66
    assert len(nodes[6].get_content()) == 94
    assert len(nodes[7].get_content()) == 100
    assert len(nodes[8].get_content()) == 69
    assert len(nodes[9].get_content()) == 95
    assert len(nodes[10].get_content()) == 77
    assert len(nodes[11].get_content()) == 114
    assert len(nodes[12].get_content()) == 65


@pytest.mark.skipif(not spacy_available, reason="Spacy model not available")
def test_chunk_size_2() -> None:
    splitter.max_chunk_size = 200
    nodes = splitter.get_nodes_from_documents([doc])
    for node in nodes:
        assert len(node.get_content()) < 200


@pytest.mark.skipif(not spacy_available, reason="Spacy model not available")
def test_chunk_size_3() -> None:
    splitter.max_chunk_size = 500
    nodes = splitter.get_nodes_from_documents([doc_same])
    for node in nodes:
        assert len(node.get_content()) < 500


def test_embed_model_path_returns_nodes() -> None:
    """With embed_model set, chunking uses embeddings instead of Spacy (no Spacy required)."""
    embed = MockEmbedding(embed_dim=4)
    splitter = SemanticDoubleMergingSplitterNodeParser.from_defaults(
        embed_model=embed,
        initial_threshold=0.6,
        appending_threshold=0.8,
        merging_threshold=0.8,
        max_chunk_size=1000,
    )
    nodes = splitter.get_nodes_from_documents([doc])
    assert len(nodes) >= 1
    assert all(len(n.get_content()) > 0 for n in nodes)


def test_embed_model_similarity_in_range() -> None:
    """_similarity with embed_model returns a value in [0, 1] (cosine-like)."""
    embed = MockEmbedding(embed_dim=4)
    splitter = SemanticDoubleMergingSplitterNodeParser.from_defaults(
        embed_model=embed,
    )
    sim = splitter._similarity("first sentence.", "second sentence.")
    assert 0 <= sim <= 1


def test_embed_model_single_sentence_document() -> None:
    """Single-sentence document yields one node when using embed_model."""
    single_doc = Document(text="Only one sentence here.")
    embed = MockEmbedding(embed_dim=4)
    splitter = SemanticDoubleMergingSplitterNodeParser.from_defaults(
        embed_model=embed,
    )
    nodes = splitter.get_nodes_from_documents([single_doc])
    assert len(nodes) == 1
    assert nodes[0].get_content() == "Only one sentence here."


class _BagOfWordsEmbedding(BaseEmbedding):
    """Deterministic embedding that records every text it is asked to embed."""

    embedded_texts: List[str] = []

    @staticmethod
    def _vec(text: str) -> List[float]:
        vec = [0.0] * 16
        for word in text.lower().split():
            vec[zlib.crc32(word.encode()) % 16] += 1.0
        return vec

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._vec(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        self.embedded_texts.append(text)
        return self._vec(text)


_TOPIC_DOC = Document(
    text=" ".join(
        f"{' '.join(words)} item{i}."
        for i, words in enumerate(
            [["cat", "dog", "pet", "fur"]] * 6
            + [["stock", "bank", "price", "trade"]] * 6
            + [["rain", "cloud", "storm", "wind"]] * 6
        )
    )
)


def test_embed_model_embeds_each_text_once() -> None:
    """Each sentence/chunk is embedded once per document, not once per comparison."""
    embed = _BagOfWordsEmbedding(embedded_texts=[], embed_dim=16)
    splitter = SemanticDoubleMergingSplitterNodeParser.from_defaults(
        embed_model=embed,
        initial_threshold=0.5,
        appending_threshold=0.5,
        merging_threshold=0.5,
        merging_range=2,
    )
    nodes = splitter.get_nodes_from_documents([_TOPIC_DOC])

    assert len(nodes) >= 1
    assert len(embed.embedded_texts) == len(set(embed.embedded_texts))


def test_similarity_reuses_cached_embeddings() -> None:
    embed = _BagOfWordsEmbedding(embedded_texts=[], embed_dim=16)
    splitter = SemanticDoubleMergingSplitterNodeParser.from_defaults(embed_model=embed)
    cache: dict = {}

    first = splitter._similarity("cat dog pet", "stock bank price", cache)
    second = splitter._similarity("stock bank price", "cat dog pet", cache)

    assert first == pytest.approx(second)
    assert embed.embedded_texts == ["cat dog pet", "stock bank price"]


def test_spacy_path_parses_each_text_once() -> None:
    """The Spacy path must not re-run the pipeline on a text it already parsed."""
    parsed: List[str] = []

    class _FakeDoc:
        def __init__(self, text: str) -> None:
            self.words = set(text.split())

        def similarity(self, other: Any) -> float:
            return len(self.words & other.words) / max(len(self.words | other.words), 1)

    def fake_nlp(text: str) -> _FakeDoc:
        parsed.append(text)
        return _FakeDoc(text)

    language_config = LanguageConfig()
    language_config.nlp = fake_nlp  # type: ignore[assignment]
    language_config.stopwords = []
    # an already loaded model must not be reloaded on every parse call
    language_config.load_model = lambda: pytest.fail("model reloaded")  # type: ignore[method-assign]
    splitter = SemanticDoubleMergingSplitterNodeParser(
        language_config=language_config,
        initial_threshold=0.5,
        appending_threshold=0.5,
        merging_threshold=0.5,
    )
    nodes = splitter.get_nodes_from_documents([_TOPIC_DOC])

    assert len(nodes) >= 1
    assert len(parsed) == len(set(parsed))


def test_clean_text_advanced() -> None:
    """Test that _clean_text_advanced properly filters out stopwords from a string."""
    from llama_index.core.utils import globals_helper

    splitter = SemanticDoubleMergingSplitterNodeParser()
    splitter.language_config.stopwords = set(globals_helper.stopwords)
    cleaned = splitter._clean_text_advanced(
        "this is a test text containing some stopwords like the and a"
    )
    assert cleaned == "test text containing stopwords like"
