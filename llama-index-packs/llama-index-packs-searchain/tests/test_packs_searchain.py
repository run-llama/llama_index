from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.searchain.base import _normalize_answer, _match_or_not
from llama_index.packs.searchain.base import _have_seen_or_not
from unittest.mock import patch, MagicMock
from llama_index.packs.searchain.base import SearChainPack, ChatMessage


def test_class():
    names_of_base_classes = [b.__name__ for b in SearChainPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes


def test_normalize_answer_simple():
    assert _normalize_answer("The Apple!") == "apple"
    assert _normalize_answer("An orange?") == "orange"
    assert _normalize_answer("  THE big BANANA.  ") == "big banana"


def test_match_or_not():
    assert _match_or_not("The big banana is tasty", "banana")
    assert not _match_or_not("This is an apple pie", "banana")


def test_have_seen_or_not_unsolved():
    mock_model = MagicMock()
    result = _have_seen_or_not(
        mock_model, "new query", ["seen one"], query_type="Unsolved Query"
    )
    assert result is False


def test_have_seen_or_not_seen_above_threshold():
    mock_model = MagicMock()
    mock_model.predict.return_value = 0.8
    result = _have_seen_or_not(
        mock_model, "new query", ["seen one"], query_type="Normal Query"
    )
    assert result is True


def test_have_seen_or_not_all_below_threshold():
    mock_model = MagicMock()
    mock_model.predict.return_value = 0.2
    result = _have_seen_or_not(
        mock_model, "new query", ["seen one"], query_type="Normal Query"
    )
    assert result is False


def test_extract_with_final_content():
    pack = SearChainPack.__new__(SearChainPack)
    messages = [
        ChatMessage(role="user", content="question"),
        ChatMessage(
            role="assistant", content="Final Content: This is the answer.\nOther line."
        ),
    ]
    result = pack._extract(messages)
    assert "Final Content" in result


def test_extract_without_final_content():
    pack = SearChainPack.__new__(SearChainPack)
    messages = [
        ChatMessage(role="user", content="question"),
        ChatMessage(role="assistant", content="No useful output."),
    ]
    result = pack._extract(messages)
    assert result == "Sorry, I still cannot solve this question!"


@patch("llama_index.packs.searchain.base.VectorStoreIndex.from_documents")
@patch("llama_index.packs.searchain.base.SimpleDirectoryReader")
@patch("llama_index.packs.searchain.base.CrossEncoder")
@patch("llama_index.packs.searchain.base.DPRReaderTokenizer.from_pretrained")
@patch("llama_index.packs.searchain.base.DPRReader.from_pretrained")
def test_searchainpack_init(
    mock_dprmodel, mock_tokenizer, mock_cross, mock_reader, mock_index
):
    mock_reader.return_value.load_data.return_value = ["doc1", "doc2"]
    mock_index.return_value.as_query_engine.return_value = MagicMock()
    instance = SearChainPack(data_path="fake_path", device="cpu")
    assert instance.device == "cpu"
    assert hasattr(instance, "llm")
    assert instance.index is not None
