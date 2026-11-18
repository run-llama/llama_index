import importlib.util

import pytest

from llama_index.core.schema import Document, MetadataMode
from llama_index.core.node_parser import (
    SimpleFileNodeParser,
)


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


FILE_FORMATS = [
    (".MD", "# First\nalpha\n# Second\nbeta", ["# First\nalpha", "# Second\nbeta"]),
    pytest.param(
        ".HTML",
        "<h1>First</h1><p>alpha</p>",
        ["First", "alpha"],
        marks=pytest.mark.skipif(
            importlib.util.find_spec("bs4") is None, reason="Requires beautifulsoup4."
        ),
    ),
    (
        ".JSON",
        '[{"answer": "alpha"}, {"answer": "beta"}]',
        ["answer alpha", "answer beta"],
    ),
]


@pytest.mark.parametrize(("extension", "text", "expected"), FILE_FORMATS)
def test_explicit_extension_is_case_insensitive(
    extension: str, text: str, expected: list[str]
) -> None:
    parser = SimpleFileNodeParser()
    for suffix in (extension, extension.lower(), extension.title()):
        document = Document(id_="source", text=text, metadata={"extension": suffix})
        nodes = parser.get_nodes_from_documents([document])
        assert [
            node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes
        ] == expected
        assert all(node.ref_doc_id == "source" for node in nodes)
        assert all(node.metadata["extension"] == suffix for node in nodes)
        assert document.metadata == {"extension": suffix}


@pytest.mark.asyncio
@pytest.mark.parametrize(("extension", "text", "expected"), FILE_FORMATS)
async def test_async_explicit_extension_is_case_insensitive(
    extension: str, text: str, expected: list[str]
) -> None:
    document = Document(text=text, metadata={"extension": extension})
    nodes = await SimpleFileNodeParser().aget_nodes_from_documents([document])
    assert [
        node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes
    ] == expected


@pytest.mark.parametrize(("extension", "text", "expected"), FILE_FORMATS)
def test_file_path_extension_is_case_insensitive(
    extension: str, text: str, expected: list[str]
) -> None:
    document = Document(text=text, metadata={"file_path": f"source{extension}"})
    nodes = SimpleFileNodeParser().get_nodes_from_documents([document])
    assert [
        node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes
    ] == expected


@pytest.mark.parametrize("extension", [".TXT", "MD", "text/markdown", "", 42, False])
def test_unknown_explicit_extension_keeps_fallback(extension: object) -> None:
    text = "# First\nalpha\n# Second\nbeta"
    document = Document(
        text=text, metadata={"extension": extension, "file_path": "source.md"}
    )
    nodes = SimpleFileNodeParser().get_nodes_from_documents([document])
    assert len(nodes) == 1
    assert nodes[0].get_content(metadata_mode=MetadataMode.NONE) == text


def test_none_extension_uses_file_path() -> None:
    document = Document(
        text="# First\nalpha\n# Second\nbeta",
        metadata={"extension": None, "file_path": "source.MD"},
    )
    nodes = SimpleFileNodeParser().get_nodes_from_documents([document])
    assert [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes] == [
        "# First\nalpha",
        "# Second\nbeta",
    ]
