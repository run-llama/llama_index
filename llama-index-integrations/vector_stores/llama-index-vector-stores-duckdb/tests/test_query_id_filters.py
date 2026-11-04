import asyncio

import pytest

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.duckdb import DuckDBVectorStore


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("node_ids", "doc_ids", "color", "expected"),
    [
        (None, None, None, ["outside"]),
        (["keep-1", "keep-2"], None, None, ["keep-1"]),
        (None, ["doc'kept"], None, ["keep-1"]),
        (["keep-2"], ["doc'kept"], None, ["keep-2"]),
        (["outside"], ["doc'kept"], None, []),
        (["keep-1", "keep-2"], None, "red", ["keep-2"]),
        (None, ["doc'kept"], "red", ["keep-2"]),
        ([], None, None, []),
        (None, [], None, []),
        (["missing"], None, None, []),
        (None, ["missing"], None, []),
    ],
)
def test_query_honors_id_constraints_before_top_k(
    use_async: bool,
    node_ids: list[str] | None,
    doc_ids: list[str] | None,
    color: str | None,
    expected: list[str],
) -> None:
    store = DuckDBVectorStore(embed_dim=2)
    store.add(
        [
            TextNode(
                id_=node_id,
                text=node_id,
                embedding=embedding,
                metadata={"color": node_color},
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)
                },
            )
            for node_id, doc_id, embedding, node_color in [
                ("outside", "other", [1.0, 0.0], "red"),
                ("keep-1", "doc'kept", [0.8, 0.6], "blue"),
                ("keep-2", "doc'kept", [0.6, 0.8], "red"),
            ]
        ]
    )
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=1,
        node_ids=node_ids,
        doc_ids=doc_ids,
        filters=MetadataFilters(filters=[MetadataFilter(key="color", value=color)])
        if color is not None
        else None,
    )

    result = asyncio.run(store.aquery(query)) if use_async else store.query(query)

    assert result.ids == expected
    assert [node.node_id for node in result.nodes] == expected
    assert len(result.similarities) == len(expected)
