import importlib.util
from typing import List, Tuple
from uuid import UUID

import pytest

from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core.schema import BaseNode, Document, MetadataMode, NodeRelationship


def test_unsupported_extension() -> None:
    simple_file_node_parser = SimpleFileNodeParser()

    nodes = simple_file_node_parser._parse_nodes(
        [
            Document(
                text="""def evenOdd(n):

  # if n&1 == 0, then num is even
  if n & 1:
    return False
  # if n&1 == 1, then num is odd
  else:
    return True"""
            )
        ]
    )
    assert len(nodes) == 1
    assert (
        nodes[0].text
        == "def evenOdd(n):\n\n  # if n&1 == 0, then num is even\n  if n & 1:\n    return False\n  # if n&1 == 1, then num is odd\n  else:\n    return True"
    )


@pytest.mark.parametrize(
    ("extension", "text", "expected_text"),
    [
        pytest.param(".md", "# Heading\nBody", "# Heading\nBody", id="markdown"),
        pytest.param(".json", '{"name":"Ada"}', "name Ada", id="json-object"),
    ],
)
def test_simple_file_parser_forwards_custom_id_func(
    extension: str, text: str, expected_text: str
) -> None:
    calls: List[Tuple[int, BaseNode]] = []

    def custom_id(index: int, source: BaseNode) -> str:
        calls.append((index, source))
        return f"custom::{source.id_}::{index}"

    document = Document(
        id_="source-document", text=text, metadata={"extension": extension}
    )
    parser = SimpleFileNodeParser(id_func=custom_id)
    assert parser.id_func is custom_id
    parsed_ids: List[str] = []

    for parse_index in range(2):
        nodes = parser.get_nodes_from_documents([document])

        assert len(nodes) == 1
        assert len(calls) == parse_index + 1
        index, source = calls[-1]
        assert index == 0
        assert source is document

        node = nodes[0]
        parsed_ids.append(node.node_id)
        assert node.get_content(metadata_mode=MetadataMode.NONE) == expected_text
        assert NodeRelationship.SOURCE in node.relationships
        source_node = node.source_node
        assert source_node is not None
        assert source_node.node_id == document.id_

    assert parsed_ids == [
        "custom::source-document::0",
        "custom::source-document::0",
    ]


@pytest.mark.parametrize(
    ("extension", "text", "expected_text"),
    [
        pytest.param(".md", "# Heading\nBody", "# Heading\nBody", id="markdown"),
        pytest.param(".json", '{"name":"Ada"}', "name Ada", id="json-object"),
    ],
)
def test_simple_file_parser_default_id(
    extension: str, text: str, expected_text: str
) -> None:
    document = Document(
        id_="source-document", text=text, metadata={"extension": extension}
    )
    nodes = SimpleFileNodeParser.from_defaults().get_nodes_from_documents([document])

    assert len(nodes) == 1
    node = nodes[0]
    assert UUID(node.node_id).version == 4
    assert node.get_content(metadata_mode=MetadataMode.NONE) == expected_text
    assert node.source_node is not None
    assert node.source_node.node_id == document.id_


@pytest.mark.parametrize(
    ("extension", "text"),
    [
        pytest.param(".txt", "Body", id="fallback"),
        pytest.param(
            ".html",
            "<p>Body</p>",
            id="html",
            marks=pytest.mark.xfail(
                raises=ImportError,
                reason="Requires beautifulsoup4.",
                condition=importlib.util.find_spec("bs4") is None,
            ),
        ),
    ],
)
def test_simple_file_parser_custom_id_for_html_and_fallback(
    extension: str, text: str
) -> None:
    calls: List[Tuple[int, BaseNode]] = []

    def custom_id(index: int, source: BaseNode) -> str:
        calls.append((index, source))
        return f"custom::{source.id_}::{index}"

    document = Document(
        id_="source-document", text=text, metadata={"extension": extension}
    )
    nodes = SimpleFileNodeParser(id_func=custom_id).get_nodes_from_documents([document])

    assert len(nodes) == 1
    assert len(calls) == 1
    index, source = calls[0]
    assert index == 0
    assert source is document
    node = nodes[0]
    assert node.node_id == "custom::source-document::0"
    assert node.get_content(metadata_mode=MetadataMode.NONE) == "Body"
    assert node.source_node is not None
    assert node.source_node.node_id == document.id_
