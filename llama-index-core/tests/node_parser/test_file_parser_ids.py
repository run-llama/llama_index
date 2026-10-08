from unittest.mock import Mock

import pytest

from llama_index.core.node_parser import HTMLNodeParser, MarkdownNodeParser
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore


@pytest.mark.parametrize(
    ("parser_cls", "text"),
    [
        (HTMLNodeParser, "<h1>First</h1><p>Second</p><h2>Third</h2>"),
        (MarkdownNodeParser, "# First\n\n## Second\n\n### Third"),
    ],
)
@pytest.mark.parametrize("include_metadata", [False, True])
def test_section_ids_preserve_all_nodes(parser_cls, text, include_metadata):
    if parser_cls is HTMLNodeParser:
        pytest.importorskip("bs4")
    documents = [Document(id_=doc_id, text=text) for doc_id in ("first", "second")]
    id_func = Mock(side_effect=lambda index, document: f"{document.id_}_{index}")
    parser = parser_cls(id_func=id_func, include_metadata=include_metadata)

    nodes = parser.get_nodes_from_documents(documents)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    assert len(docstore.docs) == 6
    assert [node.node_id for node in nodes] == [
        f"{document.id_}_{index}" for document in documents for index in range(3)
    ]
    assert [call.args[0] for call in id_func.call_args_list] == [0, 1, 2, 0, 1, 2]
    for offset, document in zip((0, 3), documents):
        for index in range(3):
            node = nodes[offset + index]
            assert node.ref_doc_id == document.id_
            if index:
                assert node.prev_node.node_id == nodes[offset + index - 1].node_id
            if index < 2:
                assert node.next_node.node_id == nodes[offset + index + 1].node_id
    assert [node.node_id for node in parser.get_nodes_from_documents(documents)] == [
        node.node_id for node in nodes
    ]
