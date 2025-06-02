import pytest

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

    assert len(nodes) == 4


@pytest.mark.skipif(not spacy_available, reason="Spacy model not available")
def test_creating_initial_chunks() -> None:
    text = doc.text
    sentences = splitter.sentence_splitter(text)
    initial_chunks = splitter._create_initial_chunks(sentences)

    assert len(initial_chunks) == 9


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
    assert len(nodes) == 9
    assert len(nodes[0].get_content()) < 200
    assert len(nodes[1].get_content()) < 200
    assert len(nodes[2].get_content()) < 200
    assert len(nodes[3].get_content()) < 200
    assert len(nodes[4].get_content()) < 200
    assert len(nodes[5].get_content()) < 200
    assert len(nodes[6].get_content()) < 200
    assert len(nodes[7].get_content()) < 200
    assert len(nodes[8].get_content()) < 200
