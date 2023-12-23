from llama_index.node_parser.relational.markdown_element import (
    MarkdownElementNodeParser,
)
from llama_index.schema import Document, IndexNode, TextNode


def test_md_table_extraction() -> None:
    test_data = Document(
        text="""
# This is a test

| Year | Benefits |
| ---- | -------- |
| 2020 | 12,000   |
| 2021 | 10,000   |
| 2022 | 130,000  |


# This is another test

## Maybe a subheader

| Year | Benefits | age | customers |
| ---- | -------- | --- | --------- |
| 2020 | 12,000   | 12  | 100       |
| 2021 | 10,000   | 13  | 200       |
| 2022 | 130,000  | 14  | 300       |

        """
    )

    node_parser = MarkdownElementNodeParser()

    nodes = node_parser.get_nodes_from_documents([test_data])
    print(f"Number of nodes: {len(nodes)}")
    for i, node in enumerate(nodes, start=0):
        print(f"Node {i}: {node}, Type: {type(node)}")
    assert len(nodes) == 6
    assert isinstance(nodes[0], TextNode)
    assert isinstance(nodes[1], IndexNode)
    assert isinstance(nodes[2], TextNode)
    assert isinstance(nodes[3], TextNode)
    assert isinstance(nodes[4], IndexNode)
    assert isinstance(nodes[5], TextNode)
