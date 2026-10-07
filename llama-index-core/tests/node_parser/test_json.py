import pytest

from llama_index.core.node_parser.file.json import JSONNodeParser
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore


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
    assert result[0].node_id != result[1].node_id


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


def test_custom_ids_preserve_array_elements_in_docstore() -> None:
    parser = JSONNodeParser(id_func=lambda i, doc: f"{doc.id_}_{i}")
    documents = [
        Document(id_="people", text='[{"name": "John"}, {"name": "Alice"}]'),
        Document(id_="other", text='{"name": "Bob"}'),
    ]
    nodes = parser.get_nodes_from_documents(documents)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    assert [node.text for node in docstore.docs.values()] == [
        "name John",
        "name Alice",
        "name Bob",
    ]
    assert [node.node_id for node in nodes] == ["people_0", "people_1", "other_0"]
    assert [node.ref_doc_id for node in nodes] == ["people", "people", "other"]
    assert nodes[0].next_node is not None
    assert nodes[0].next_node.node_id == "people_1"
    assert nodes[1].prev_node is not None
    assert nodes[1].prev_node.node_id == "people_0"
    assert nodes[1].next_node is None
    assert nodes[2].prev_node is None

    repeated_nodes = parser.get_nodes_from_documents(documents)
    assert [node.node_id for node in repeated_nodes] == [node.node_id for node in nodes]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[]", []),
        ('{"name": "Alice"}', ["name Alice"]),
        ('[[1, 2], {"nested": {"value": 3}}]', ["1\n2", "nested value 3"]),
    ],
)
def test_custom_ids_preserve_json_splits(text: str, expected: list[str]) -> None:
    parser = JSONNodeParser(id_func=lambda i, doc: f"{doc.id_}_{i}")
    nodes = parser.get_nodes_from_documents([Document(id_="json", text=text)])
    assert [node.text for node in nodes] == expected
    assert [node.node_id for node in nodes] == [
        f"json_{i}" for i in range(len(expected))
    ]
