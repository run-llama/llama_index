"""Node postprocessor tests."""

import pytest

from gpt_index.data_structs.node_v2 import Node, DocumentRelationship
from gpt_index.indices.postprocessor.node import PrevNextNodePostprocessor
from gpt_index.docstore import DocumentStore


def test_forward_back_processor() -> None:
    """Test forward-back processor."""

    nodes = [
        Node("Hello world.", doc_id="1"),
        Node("This is a test.", doc_id="2"),
        Node("This is another test.", doc_id="3"),
        Node("This is a test v2.", doc_id="4"),
    ]
    for i, node in enumerate(nodes):
        if i > 0:
            node.relationships.update(
                {DocumentRelationship.PREVIOUS: nodes[i - 1].get_doc_id()},
            )
        if i < len(nodes) - 1:
            node.relationships.update(
                {DocumentRelationship.NEXT: nodes[i + 1].get_doc_id()},
            )

    docstore = DocumentStore()
    docstore.add_documents(nodes)

    # check for a single node
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=2, mode="next"
    )
    processed_nodes = node_postprocessor.postprocess_nodes([nodes[0]])
    assert len(processed_nodes) == 3
    assert processed_nodes[0].get_doc_id() == "1"
    assert processed_nodes[1].get_doc_id() == "2"
    assert processed_nodes[2].get_doc_id() == "3"

    # check for multiple nodes (nodes should not be duped)
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=1, mode="next"
    )
    processed_nodes = node_postprocessor.postprocess_nodes([nodes[1], nodes[2]])
    assert len(processed_nodes) == 3
    assert processed_nodes[0].get_doc_id() == "2"
    assert processed_nodes[1].get_doc_id() == "3"
    assert processed_nodes[2].get_doc_id() == "4"

    # check for previous
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=1, mode="previous"
    )
    processed_nodes = node_postprocessor.postprocess_nodes([nodes[1], nodes[2]])
    assert len(processed_nodes) == 3
    assert processed_nodes[0].get_doc_id() == "1"
    assert processed_nodes[1].get_doc_id() == "2"
    assert processed_nodes[2].get_doc_id() == "3"

    # check that both works
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=1, mode="both"
    )
    processed_nodes = node_postprocessor.postprocess_nodes([nodes[2]])
    assert len(processed_nodes) == 3
    # nodes are sorted
    assert processed_nodes[0].get_doc_id() == "2"
    assert processed_nodes[1].get_doc_id() == "3"
    assert processed_nodes[2].get_doc_id() == "4"

    # check that num_nodes too high still works
    node_postprocessor = PrevNextNodePostprocessor(
        docstore=docstore, num_nodes=4, mode="both"
    )
    processed_nodes = node_postprocessor.postprocess_nodes([nodes[2]])
    assert len(processed_nodes) == 4
    # nodes are sorted
    assert processed_nodes[0].get_doc_id() == "1"
    assert processed_nodes[1].get_doc_id() == "2"
    assert processed_nodes[2].get_doc_id() == "3"
    assert processed_nodes[3].get_doc_id() == "4"

    # check that raises value error for invalid mode
    with pytest.raises(ValueError):
        PrevNextNodePostprocessor(docstore=docstore, num_nodes=4, mode="asdfasdf")
