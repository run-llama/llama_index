from unittest.mock import PropertyMock, patch

from llama_index.core.node_parser.file.json import JSONNodeParser
from llama_index.core.schema import Document


def test_split_empty_text() -> None:
    json_splitter = JSONNodeParser()
    input_text = Document(text="")
    result = json_splitter.get_nodes_from_documents([input_text])
    assert result == []


def test_split_valid_json() -> None:
    json_splitter = JSONNodeParser()
    input_text = Document(
        text='[{"name": "John", "age": 30}, {"name": "Alice", "age": 25}]'
    )
    result = json_splitter.get_nodes_from_documents([input_text])
    assert len(result) == 2
    assert result[0].text == "name John\nage 30"
    assert result[1].text == "name Alice\nage 25"


def test_split_valid_json_defaults() -> None:
    json_splitter = JSONNodeParser()
    input_text = Document(text='[{"name": "John", "age": 30}]')
    result = json_splitter.get_nodes_from_documents([input_text])
    assert len(result) == 1
    assert result[0].text == "name John\nage 30"


def test_split_valid_dict_json() -> None:
    json_splitter = JSONNodeParser()
    input_text = Document(text='{"name": "John", "age": 30}')
    result = json_splitter.get_nodes_from_documents([input_text])
    assert len(result) == 1
    assert result[0].text == "name John\nage 30"


def test_split_invalid_json() -> None:
    json_splitter = JSONNodeParser()
    input_text = Document(text='{"name": "John", "age": 30,}')
    result = json_splitter.get_nodes_from_documents([input_text])
    assert result == []


def test_source_document_hashed_once_per_document() -> None:
    input_text = Document(text="[" + ",".join('{"a": %d}' % i for i in range(50)) + "]")
    with patch.object(Document, "hash", new_callable=PropertyMock) as mock_hash:
        mock_hash.return_value = "doc-hash"
        result = JSONNodeParser().get_nodes_from_documents([input_text])

    assert len(result) == 50
    assert mock_hash.call_count == 1
