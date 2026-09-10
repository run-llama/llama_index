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


def test_split_scalar_json_raises_clear_error() -> None:
    """Valid JSON that is not an object/array raises a descriptive error."""
    import pytest

    json_splitter = JSONNodeParser()
    with pytest.raises(ValueError, match="must be an object or an array"):
        json_splitter.get_nodes_from_documents([Document(text="5")])


def test_split_empty_object_yields_no_nodes() -> None:
    """An empty object yields no nodes, consistent with an empty array."""
    json_splitter = JSONNodeParser()
    assert json_splitter.get_nodes_from_documents([Document(text="{}")]) == []
    assert json_splitter.get_nodes_from_documents([Document(text='[{}, {"a": 1}]')])[0].text == "a 1"
