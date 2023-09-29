from llama_index.node_parser.file.json import JSONNodeParser
from llama_index.schema import Document


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
    assert "name John\nage 30" == result[0].text
    assert "name Alice\nage 25" == result[1].text


def test_split_valid_json_defaults() -> None:
    json_splitter = JSONNodeParser()
    input_text = Document(text='[{"name": "John", "age": 30}]')
    result = json_splitter.get_nodes_from_documents([input_text])
    assert len(result) == 1
    assert "name John\nage 30" == result[0].text


def test_split_valid_dict_json() -> None:
    json_splitter = JSONNodeParser()
    input_text = Document(text='{"name": "John", "age": 30}')
    result = json_splitter.get_nodes_from_documents([input_text])
    assert len(result) == 1
    assert "name John\nage 30" == result[0].text


def test_split_invalid_json() -> None:
    json_splitter = JSONNodeParser()
    input_text = Document(text='{"name": "John", "age": 30,}')
    result = json_splitter.get_nodes_from_documents([input_text])
    assert result == []
