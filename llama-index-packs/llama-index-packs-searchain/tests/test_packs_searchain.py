from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.searchain import SearChainPack
from llama_index.packs.searchain.base import _normalize_answer, _match_or_not
from unittest.mock import patch, MagicMock
from llama_index.packs.searchain.base import SearChainPack


def test_class():
    names_of_base_classes = [b.__name__ for b in SearChainPack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes


def test_normalize_answer():
    assert _normalize_answer("  The Apple!! ") == "apple"


def test_match_or_not():
    assert _match_or_not("The Apple Pie", "apple")
    assert not _match_or_not("Banana", "apple")


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
