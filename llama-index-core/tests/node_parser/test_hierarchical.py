from typing import Any

import pytest

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_child_nodes,
    get_deeper_nodes,
    get_leaf_nodes,
    get_root_nodes,
)

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


def test_async_entry_points_match_sync() -> None:
    """
    Async entry points must build the same hierarchy as the sync ones.

    aget_nodes_from_documents, acall and IngestionPipeline.arun previously
    returned the input documents unchanged because the async path went through
    the identity _parse_nodes implementation.
    See https://github.com/run-llama/llama_index/issues/23444.
    """
    from llama_index.core.async_utils import asyncio_run
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser.text.sentence import SentenceSplitter

    def det_id(i: int, parent: Any) -> str:
        return f"{parent.id_}-{i}"

    docs = [Document.example()]
    parser = HierarchicalNodeParser(
        node_parser_ids=["chunk_size_512", "chunk_size_128"],
        node_parser_map={
            "chunk_size_512": SentenceSplitter(
                chunk_size=512, chunk_overlap=20, id_func=det_id
            ),
            "chunk_size_128": SentenceSplitter(
                chunk_size=128, chunk_overlap=20, id_func=det_id
            ),
        },
    )

    sync_nodes = parser.get_nodes_from_documents(docs)
    async_nodes = asyncio_run(parser.aget_nodes_from_documents(docs))
    acall_nodes = asyncio_run(parser.acall(docs))
    pipeline_nodes = asyncio_run(
        IngestionPipeline(transformations=[parser]).arun(documents=docs)
    )

    # More than one node and a real hierarchy (leaf nodes differ from input).
    assert len(sync_nodes) > 1
    assert len(get_leaf_nodes(sync_nodes)) > 1

    for result in (async_nodes, acall_nodes, pipeline_nodes):
        assert result == sync_nodes
