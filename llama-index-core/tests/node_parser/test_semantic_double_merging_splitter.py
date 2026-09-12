import pytest

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


def test_clean_text_advanced() -> None:
    """Test that _clean_text_advanced properly filters out stopwords from a string."""
    from llama_index.core.utils import globals_helper

    splitter = SemanticDoubleMergingSplitterNodeParser()
    splitter.language_config.stopwords = set(globals_helper.stopwords)
    cleaned = splitter._clean_text_advanced(
        "this is a test text containing some stopwords like the and a"
    )
    assert cleaned == "test text containing stopwords like"


class _AlwaysSimilarSplitter(SemanticDoubleMergingSplitterNodeParser):
    """Similarity is pinned so only the chunk-size arithmetic decides where chunks end."""

    def _similarity(self, text_a: str, text_b: str) -> float:
        return 1.0


def test_multi_character_merging_separator_respects_max_chunk_size() -> None:
    """A separator longer than one character must not push merged chunks over max_chunk_size."""
    max_chunk_size = 100
    separator = "\n\n---\n\n"

    splitter = _AlwaysSimilarSplitter(
        max_chunk_size=max_chunk_size, merging_separator=separator, merging_range=1
    )

    # 50 + 49 + len(" ") == 100, so the old guard treated this as a fit.
    merged = splitter._merge_initial_chunks(["A" * 50, "B" * 49])
    assert max(len(chunk) for chunk in merged) <= max_chunk_size

    created = splitter._create_initial_chunks(["A" * 50, "B" * 20, "C" * 20])
    assert max(len(chunk) for chunk in created) <= max_chunk_size


@pytest.mark.parametrize("separator_len", [1, 2, 4, 8, 16])
def test_merging_never_exceeds_max_chunk_size(separator_len: int) -> None:
    """Sweep the boundary for a range of separator lengths."""
    max_chunk_size = 100
    splitter = _AlwaysSimilarSplitter(
        max_chunk_size=max_chunk_size,
        merging_separator="s" * separator_len,
        merging_range=1,
    )

    for first in range(1, max_chunk_size - 1):
        second = max_chunk_size - first - 1
        merged = splitter._merge_initial_chunks(["A" * first, "B" * second])
        assert max(len(chunk) for chunk in merged) <= max_chunk_size
