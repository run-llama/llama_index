"""Test for duplicate text position bug fix."""
from typing import Any, List, Sequence
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, TextNode, Document, NodeRelationship


class _TestNodeParser(NodeParser):
    """Minimal NodeParser implementation for testing."""
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return list(nodes)


def test_postprocess_handles_duplicate_text_correctly():
    """
    Test that nodes with identical text are assigned to different positions.
    
    This tests the fix for a bug where parent_doc.text.find() would always
    return the first occurrence, causing all nodes with identical text to
    have the same start/end positions.
    """
    # Create a document with repeated text
    doc_text = """Chapter 1: Introduction
This is important.

Chapter 2: Methods
This is important.

Chapter 3: Results
This is important."""
    
    doc = Document(text=doc_text, doc_id="test_doc")
    
    # Create nodes with identical text that appear at different positions
    # These represent what a parser might return after splitting the document
    nodes = []
    
    # Node 1: First occurrence of "This is important."
    node1 = TextNode(text="This is important.")
    node1.relationships = {NodeRelationship.SOURCE: doc.as_related_node_info()}
    nodes.append(node1)
    
    # Node 2: Second occurrence of "This is important."
    node2 = TextNode(text="This is important.")
    node2.relationships = {NodeRelationship.SOURCE: doc.as_related_node_info()}
    nodes.append(node2)
    
    # Node 3: Third occurrence of "This is important."
    node3 = TextNode(text="This is important.")
    node3.relationships = {NodeRelationship.SOURCE: doc.as_related_node_info()}
    nodes.append(node3)
    
    # Run postprocessing
    np = _TestNodeParser()
    processed_nodes = np._postprocess_parsed_nodes(nodes, {doc.doc_id: doc})
    
    # Verify that each node has unique positions
    assert isinstance(processed_nodes[0], TextNode)
    assert processed_nodes[0].start_char_idx == 24  # First "This is important."
    assert processed_nodes[0].end_char_idx == 42
    
    assert isinstance(processed_nodes[1], TextNode)
    assert processed_nodes[1].start_char_idx == 63  # Second "This is important."
    assert processed_nodes[1].end_char_idx == 81
    
    assert isinstance(processed_nodes[2], TextNode)
    assert processed_nodes[2].start_char_idx == 102  # Third "This is important."
    assert processed_nodes[2].end_char_idx == 120
    
    # Verify all positions are different
    positions = [
        (n.start_char_idx, n.end_char_idx) 
        for n in processed_nodes 
        if isinstance(n, TextNode)
    ]
    assert len(set(positions)) == 3, "All nodes should have unique positions"


def test_postprocess_handles_mixed_duplicate_and_unique_text():
    """Test with a mix of duplicate and unique text."""
    doc_text = "Header A\nUnique content here.\nHeader B\nUnique content here.\nFooter"
    doc = Document(text=doc_text, doc_id="test_doc")
    
    nodes = []
    
    # First node with unique text
    node1 = TextNode(text="Header A")
    node1.relationships = {NodeRelationship.SOURCE: doc.as_related_node_info()}
    nodes.append(node1)
    
    # Second node with text that appears twice
    node2 = TextNode(text="Unique content here.")
    node2.relationships = {NodeRelationship.SOURCE: doc.as_related_node_info()}
    nodes.append(node2)
    
    # Third node with unique text
    node3 = TextNode(text="Header B")
    node3.relationships = {NodeRelationship.SOURCE: doc.as_related_node_info()}
    nodes.append(node3)
    
    # Fourth node with duplicate text (second occurrence)
    node4 = TextNode(text="Unique content here.")
    node4.relationships = {NodeRelationship.SOURCE: doc.as_related_node_info()}
    nodes.append(node4)
    
    # Fifth node with unique text
    node5 = TextNode(text="Footer")
    node5.relationships = {NodeRelationship.SOURCE: doc.as_related_node_info()}
    nodes.append(node5)
    
    # Run postprocessing
    np = _TestNodeParser()
    processed_nodes = np._postprocess_parsed_nodes(nodes, {doc.doc_id: doc})
    
    # Verify positions
    assert isinstance(processed_nodes[0], TextNode)
    assert processed_nodes[0].start_char_idx == 0  # "Header A"
    assert processed_nodes[0].end_char_idx == 8
    
    assert isinstance(processed_nodes[1], TextNode)
    assert processed_nodes[1].start_char_idx == 9  # First "Unique content here."
    assert processed_nodes[1].end_char_idx == 29
    
    assert isinstance(processed_nodes[2], TextNode)
    assert processed_nodes[2].start_char_idx == 30  # "Header B"
    assert processed_nodes[2].end_char_idx == 38
    
    assert isinstance(processed_nodes[3], TextNode)
    assert processed_nodes[3].start_char_idx == 39  # Second "Unique content here."
    assert processed_nodes[3].end_char_idx == 59
    
    assert isinstance(processed_nodes[4], TextNode)
    assert processed_nodes[4].start_char_idx == 60  # "Footer"
    assert processed_nodes[4].end_char_idx == 66
    
    # Verify the two "Unique content here." nodes have different positions
    assert processed_nodes[1].start_char_idx != processed_nodes[3].start_char_idx


def test_postprocess_handles_multiple_documents():
    """Test that position tracking is independent per document."""
    doc1_text = "Same text here.\nSame text here."
    doc2_text = "Same text here.\nSame text here."
    
    doc1 = Document(text=doc1_text, doc_id="doc1")
    doc2 = Document(text=doc2_text, doc_id="doc2")
    
    nodes = []
    
    # Nodes from doc1
    node1 = TextNode(text="Same text here.")
    node1.relationships = {NodeRelationship.SOURCE: doc1.as_related_node_info()}
    nodes.append(node1)
    
    node2 = TextNode(text="Same text here.")
    node2.relationships = {NodeRelationship.SOURCE: doc1.as_related_node_info()}
    nodes.append(node2)
    
    # Nodes from doc2
    node3 = TextNode(text="Same text here.")
    node3.relationships = {NodeRelationship.SOURCE: doc2.as_related_node_info()}
    nodes.append(node3)
    
    node4 = TextNode(text="Same text here.")
    node4.relationships = {NodeRelationship.SOURCE: doc2.as_related_node_info()}
    nodes.append(node4)
    
    # Run postprocessing
    np = _TestNodeParser()
    processed_nodes = np._postprocess_parsed_nodes(
        nodes, 
        {doc1.doc_id: doc1, doc2.doc_id: doc2}
    )
    
    # Doc1 nodes should have different positions
    assert isinstance(processed_nodes[0], TextNode)
    assert isinstance(processed_nodes[1], TextNode)
    assert processed_nodes[0].start_char_idx == 0
    assert processed_nodes[1].start_char_idx == 16
    
    # Doc2 nodes should also have different positions (independent tracking)
    assert isinstance(processed_nodes[2], TextNode)
    assert isinstance(processed_nodes[3], TextNode)
    assert processed_nodes[2].start_char_idx == 0
    assert processed_nodes[3].start_char_idx == 16
    
    # Nodes from same document should have different positions
    assert processed_nodes[0].start_char_idx != processed_nodes[1].start_char_idx
    assert processed_nodes[2].start_char_idx != processed_nodes[3].start_char_idx

