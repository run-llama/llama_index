import pytest
from typing import List
import importlib.util

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.vector_stores.duckdb import DuckDBVectorStore


def test_duckdb_installed():
    assert importlib.util.find_spec("duckdb") is not None


@pytest.fixture(scope="module")
def text_node_list() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "author": "Stephen King",
                "theme": "Friendship",
            },
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "director": "Christopher Nolan",
            },
            embedding=[0.0, 0.0, 1.0],
        ),
        TextNode(
            text="I was taught that the way of progress was neither swift nor easy.",
            id_="0b31ae71-b797-4e88-8495-031371a7752e",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-3")},
            metadata={
                "author": "Marie Curie",
            },
            embedding=[0.0, 0.0, 0.9],
        ),
        TextNode(
            text=(
                "The important thing is not to stop questioning."
                + " Curiosity has its own reason for existing."
            ),
            id_="bd2e080b-159a-4030-acc3-d98afd2ba49b",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-4")},
            metadata={
                "author": "Albert Einstein",
            },
            embedding=[0.0, 0.0, 0.5],
        ),
        TextNode(
            text=(
                "I am no bird; and no net ensnares me;"
                + " I am a free human being with an independent will."
            ),
            id_="f658de3b-8cef-4d1c-8bed-9a263c907251",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-5")},
            metadata={
                "author": "Charlotte Bronte",
            },
            embedding=[0.0, 0.0, 0.3],
        ),
        TextNode(
            text=("中文用户来了。"),
            id_="943ef4e5-b5bc-4b85-b0d7-bc4fb25417db",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-6")},
            metadata={
                "author": "Ava Wilson",
            },
            embedding=[0.3, 0.3, 0.3],
        ),
        TextNode(
            text=(
                "Vector stores contain embedding vectors of ingested document chunks (and sometimes the document chunks as well)."
            ),
            id_="e8f0c6cb-8d35-4240-a60a-b57070b3960f",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-7")},
            metadata={
                "author": "Emma Johnson",
            },
            embedding=[0.9, 0.3, 0.3],
            excluded_embed_metadata_keys=["excluded_embed"],
            excluded_llm_metadata_keys=["excluded_llm", "metadata", "keys"],
        ),
    ]


@pytest.fixture(scope="module")
def vector_store() -> DuckDBVectorStore:
    return DuckDBVectorStore(embed_dim=3)


def test_instance_creation_from_memory(
    vector_store: DuckDBVectorStore,
) -> None:
    assert isinstance(vector_store, DuckDBVectorStore)
    assert vector_store.database_name == ":memory:"


def test_duckdb_add_and_query(
    vector_store: DuckDBVectorStore, text_node_list: List[TextNode]
) -> None:
    vector_store.add(text_node_list)

    res = vector_store.query(
        VectorStoreQuery(query_embedding=[1.1, 0.0, 0.0], similarity_top_k=1)
    )
    assert res.nodes
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"
    assert res.nodes[0].get_content() == "lorem ipsum"
    assert res.nodes[0].metadata.get("author") == "Stephen King"
    assert res.nodes[0].metadata.get("theme") == "Friendship"
    assert res.nodes[0].excluded_embed_metadata_keys == []
    assert res.nodes[0].excluded_llm_metadata_keys == []
    assert res.nodes[0].source_node.node_id == "test-0"

    res = vector_store.query(
        VectorStoreQuery(query_embedding=[0.9, 0.3, 0.3], similarity_top_k=1)
    )
    assert res.nodes
    assert res.nodes[0].node_id == "e8f0c6cb-8d35-4240-a60a-b57070b3960f"
    assert (
        res.nodes[0].get_content()
        == "Vector stores contain embedding vectors of ingested document chunks (and sometimes the document chunks as well)."
    )
    assert res.nodes[0].metadata.get("author") == "Emma Johnson"
    assert res.nodes[0].excluded_embed_metadata_keys == ["excluded_embed"]
    assert res.nodes[0].excluded_llm_metadata_keys == [
        "excluded_llm",
        "metadata",
        "keys",
    ]
    assert res.nodes[0].source_node.node_id == "test-7"
