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
