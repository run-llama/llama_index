import pytest

from llama_index.core.schema import Document
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


@pytest.mark.parametrize(
    ("extension", "text"),
    [
        (".md", "# Heading\nBody"),
        (".json", '{"question": "answer"}'),
        (".html", "<p>HTML body</p>"),
    ],
)
def test_supported_extension_uses_custom_id_func(
    extension: str, text: str
) -> None:
    calls: list[tuple[int, Document]] = []

    def custom_id(index: int, source: Document) -> str:
        calls.append((index, source))
        return f"{source.id_}::{index}"

    document = Document(
        id_="source", text=text, metadata={"extension": extension}
    )
    nodes = SimpleFileNodeParser(id_func=custom_id).get_nodes_from_documents([document])

    assert len(nodes) == 1
    assert nodes[0].node_id == "source::0"
    assert calls == [(0, document)]
