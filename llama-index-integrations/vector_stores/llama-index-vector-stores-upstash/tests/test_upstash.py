import os
from importlib.util import find_spec
from typing import List

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from llama_index.vector_stores.upstash import UpstashVectorStore

try:
    find_spec("upstash-vector")
    if os.environ.get("UPSTASH_VECTOR_URL") and os.environ.get("UPSTASH_VECTOR_TOKEN"):
        upstash_installed = True
    else:
        upstash_installed = False
except ImportError:
    upstash_installed = False


@pytest.fixture()
def upstash_vector_store() -> UpstashVectorStore:
    return UpstashVectorStore(
        url=os.environ.get("UPSTASH_VECTOR_URL") or "",
        token=os.environ.get("UPSTASH_VECTOR_TOKEN") or "",
    )


@pytest.fixture()
def text_nodes() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "author": "Stephen King",
                "theme": "Friendship",
                "rating": 4.1,
            },
            embedding=[1.0, 0.0, 0.0] * 512,
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
                "rating": 3.3,
            },
            embedding=[0.0, 1.0, 0.0] * 512,
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "director": "Christopher Nolan",
                "rating": 4.3,
                "theme": "Action",
            },
            embedding=[0.0, 0.0, 1.0] * 512,
        ),
        TextNode(
            text="I was taught that the way of progress was neither swift nor easy.",
            id_="0b31ae71-b797-4e88-8495-031371a7752e",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-3")},
            metadata={
                "author": "Marie Curie",
                "rating": 2.3,
            },
            embedding=[0.0, 0.0, 0.9] * 512,
        ),
        TextNode(
            text=(
                "The important thing is not to stop questioning."
                + " Curiosity has its own reason for existing."
            ),
            id_="bd2e080b-159a-4030-acc3-d98afd2ba49b",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-4")},
            metadata={
                "author": "Albert Einstein",
                "rating": 4.8,
            },
            embedding=[0.0, 0.0, 0.5] * 512,
        ),
        TextNode(
            text=(
                "I am no bird; and no net ensnares me;"
                + " I am a free human being with an independent will."
            ),
            id_="f658de3b-8cef-4d1c-8bed-9a263c907251",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-5")},
            metadata={
                "author": "Charlotte Bronte",
                "rating": 1.5,
            },
            embedding=[0.0, 0.0, 0.3] * 512,
        ),
    ]


@pytest.mark.skipif(not upstash_installed, reason="upstash-vector not installed")
def test_upstash_vector_add(
    upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode]
) -> None:
    res = upstash_vector_store.add(nodes=text_nodes)
    assert res == [
        "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
        "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
        "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        "0b31ae71-b797-4e88-8495-031371a7752e",
        "bd2e080b-159a-4030-acc3-d98afd2ba49b",
        "f658de3b-8cef-4d1c-8bed-9a263c907251",
    ]


@pytest.mark.skipif(not upstash_installed, reason="upstash-vector not installed")
def test_upstash_vector_query(
    upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode]
) -> None:
    upstash_vector_store.add(nodes=text_nodes)
    res = upstash_vector_store.query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0] * 512,
            similarity_top_k=1,
        )
    )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"
    # assert res.nodes and res.nodes[0].id_ in ["test_node_1", "test_node_2"]


@pytest.mark.skipif(not upstash_installed, reason="upstash-vector not installed")
def test_upstash_vector_filtering_eq(
    upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode]
) -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="author", value="Marie Curie", operator=FilterOperator.EQ
            )
        ],
    )
    upstash_vector_store.add(nodes=text_nodes)
    res = upstash_vector_store.query(
        VectorStoreQuery(
            query_embedding=[0.1] * 1536,
            filters=filters,
            similarity_top_k=1,
        )
    )
    assert len(res.nodes) == 1
    assert (
        res.nodes[0].get_content()
        == "I was taught that the way of progress was neither swift nor easy."
    )


@pytest.mark.skipif(not upstash_installed, reason="upstash-vector not installed")
def test_upstash_vector_filtering_gte(
    upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode]
) -> None:
    filters = MetadataFilters(
        filters=[MetadataFilter(key="rating", value=4.3, operator=FilterOperator.GTE)],
    )
    upstash_vector_store.add(nodes=text_nodes)
    res = upstash_vector_store.query(
        VectorStoreQuery(
            query_embedding=[0.1] * 1536,
            filters=filters,
        )
    )
    assert res.nodes
    for node in res.nodes:
        assert node.metadata["rating"] >= 4.3


@pytest.mark.skipif(not upstash_installed, reason="upstash-vector not installed")
def test_upstash_vector_filtering_in(
    upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode]
) -> None:
    values_contained = ["Friendship", "Mafia"]

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="theme", value=values_contained, operator=FilterOperator.IN
            )
        ],
    )
    upstash_vector_store.add(nodes=text_nodes)
    res = upstash_vector_store.query(
        VectorStoreQuery(
            query_embedding=[0.1] * 1536,
            filters=filters,
        )
    )
    assert res.nodes

    for node in res.nodes:
        assert node.metadata["theme"] in values_contained


@pytest.mark.skipif(not upstash_installed, reason="upstash-vector not installed")
def test_upstash_vector_filtering_composite(
    upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode]
) -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rating", value=3, operator=FilterOperator.LT),
            MetadataFilter(
                key="author", value="Charlotte Bronte", operator=FilterOperator.EQ
            ),
        ],
        condition=FilterCondition.AND,
    )
    upstash_vector_store.add(nodes=text_nodes)
    res = upstash_vector_store.query(
        VectorStoreQuery(
            query_embedding=[0.1] * 1536,
            filters=filters,
        )
    )
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "f658de3b-8cef-4d1c-8bed-9a263c907251"
