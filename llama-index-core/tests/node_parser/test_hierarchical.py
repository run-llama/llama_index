import pytest

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_child_nodes,
    get_deeper_nodes,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import MetadataMode, NodeRelationship, TextNode

ROOT_NODES_LEN = 1
CHILDREN_NODES_LEN = 3
GRAND_CHILDREN_NODES_LEN = 7


@pytest.fixture(scope="module")
def nodes() -> list:
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[512, 128, 64],
        chunk_overlap=10,
    )
    return node_parser.get_nodes_from_documents([Document.example()])


def test_get_root_nodes(nodes: list) -> None:
    root_nodes = get_root_nodes(nodes)
    assert len(root_nodes) == ROOT_NODES_LEN


def test_get_root_nodes_empty(nodes: list) -> None:
    root_nodes = get_root_nodes(get_leaf_nodes(nodes))
    assert root_nodes == []


def test_get_leaf_nodes(nodes: list) -> None:
    leaf_nodes = get_leaf_nodes(nodes)
    assert len(leaf_nodes) == GRAND_CHILDREN_NODES_LEN


def test_get_child_nodes(nodes: list) -> None:
    child_nodes = get_child_nodes(get_root_nodes(nodes), all_nodes=nodes)
    assert len(child_nodes) == CHILDREN_NODES_LEN


def test_get_deeper_nodes(nodes: list) -> None:
    deep_nodes = get_deeper_nodes(nodes, depth=0)
    assert deep_nodes == get_root_nodes(nodes)

    deep_nodes = get_deeper_nodes(nodes, depth=1)
    assert deep_nodes == get_child_nodes(get_root_nodes(nodes), nodes)

    deep_nodes = get_deeper_nodes(nodes, depth=2)
    assert deep_nodes == get_leaf_nodes(nodes)

    deep_nodes = get_deeper_nodes(nodes, depth=2)
    assert deep_nodes == get_child_nodes(
        get_child_nodes(get_root_nodes(nodes), nodes), nodes
    )


def test_get_deeper_nodes_with_no_root_nodes(nodes: list) -> None:
    with pytest.raises(ValueError, match="There is no*"):
        get_deeper_nodes(get_leaf_nodes(nodes))


def test_get_deeper_nodes_with_negative_depth(nodes: list) -> None:
    with pytest.raises(ValueError, match="Depth cannot be*"):
        get_deeper_nodes(nodes, -1)


def test_hierarchical_nodes_have_sibling_relationships_and_document_offsets() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    doc = Document(text=text, doc_id="doc1")
    nodes = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128], chunk_overlap=0
    ).get_nodes_from_documents([doc])
    non_root = [node for node in nodes if NodeRelationship.PARENT in node.relationships]

    assert sum(NodeRelationship.NEXT in node.relationships for node in non_root) == sum(
        NodeRelationship.PREVIOUS in node.relationships for node in non_root
    )
    for node in non_root:
        if isinstance(node, TextNode):
            assert node.start_char_idx is not None
            assert node.end_char_idx is not None
            assert text[node.start_char_idx : node.end_char_idx] == node.get_content(
                metadata_mode=MetadataMode.NONE
            )


def test_pipeline_resplitting_preserves_sibling_relationships_and_offsets() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    doc = Document(text=text, doc_id="doc1")
    nodes = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=0),
            TokenTextSplitter(chunk_size=128, chunk_overlap=0),
        ]
    ).run(documents=[doc])

    assert sum(NodeRelationship.NEXT in node.relationships for node in nodes) == sum(
        NodeRelationship.PREVIOUS in node.relationships for node in nodes
    )
    for node in nodes:
        assert isinstance(node, TextNode)
        assert node.start_char_idx is not None
        assert node.end_char_idx is not None
        assert text[node.start_char_idx : node.end_char_idx] == node.get_content(
            metadata_mode=MetadataMode.NONE
        )
