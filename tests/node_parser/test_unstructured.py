from llama_index.node_parser.relational.unstructured_element import (
    UnstructuredElementNodeParser,
)
from llama_index.schema import Document, IndexNode, TextNode


def test_html_table_extraction() -> None:
    test_data = Document(
        text="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <table>
            <tr>
                <td>My title center</td>
            </tr>
            <tr>
                <td>Design Website like its 2000</td>
                <td>Yeah!</td>
            </tr>
        </table>
        <p>
            Test paragraph
        </p>
        <table>
            <tr>
                <td>Year</td>
                <td>Benefits</td>
            </tr>
            <tr>
               <td>2020</td>
                <td>12,000</td>
            </tr>
            <tr>
               <td>2021</td>
                <td>10,000</td>
            </tr>
            <tr>
               <td>2022</td>
                <td>130,000</td>
            </tr>
        </table>
    </body>
    </html>
        """
    )

    node_parser = UnstructuredElementNodeParser()

    nodes = node_parser.get_nodes_from_documents([test_data])

    assert len(nodes) == 3
    assert isinstance(nodes[0], TextNode)
    assert isinstance(nodes[1], IndexNode)
    assert isinstance(nodes[2], TextNode)
