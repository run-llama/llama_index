"""Test Azure CosmosDB NoSql Vector Search functionality."""

from __future__ import annotations

import os
from time import sleep
from typing import List

import pytest

try:
    from azure.cosmos import CosmosClient, PartitionKey

    URI = os.environ.get("AZURE_COSMOSDB_URI", "")
    KEY = os.environ.get("AZURE_COSMOSDB_KEY", "")
    database_name = "test_database"
    container_name = "test_container"
    test_client = CosmosClient(URI, credential=KEY)

    indexing_policy = {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
    }

    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": 3,
            }
        ]
    }

    partition_key = PartitionKey(path="/id")
    cosmos_container_properties_test = {"partition_key": partition_key}
    cosmos_database_properties_test = {}

    # Full text search policies
    full_text_indexing_policy = {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
        "fullTextIndexes": [{"path": "/text"}, {"path": "/description"}],
    }

    full_text_policy = {
        "defaultLanguage": "en-US",
        "fullTextPaths": [
            {"path": "/text", "language": "en-US"},
            {"path": "/description", "language": "en-US"},
        ],
    }

    cosmos_container_properties_full_text = {
        "partition_key": partition_key,
        "full_text_policy": full_text_policy,
    }

    test_database = test_client.create_database_if_not_exists(id=database_name)

    # Always recreate containers to ensure the vector embedding policy (dimensions) is current
    for _cname in [container_name, "full_text_container"]:
        try:
            test_database.delete_container(_cname)
        except Exception:
            pass

    test_container = test_database.create_container(
        id=container_name,
        partition_key=partition_key,
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
    )
    full_text_container = test_database.create_container(
        id="full_text_container",
        partition_key=partition_key,
        indexing_policy=full_text_indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
        full_text_policy=full_text_policy,
    )

    cosmosnosql_available = True
except (ImportError, Exception):
    cosmosnosql_available = False

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.azurecosmosnosql import AzureCosmosDBNoSqlVectorSearch


@pytest.fixture(scope="session")
def node_embeddings() -> list[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="node-1",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")},
            metadata={
                "author": "Stephen King",
                "theme": "Friendship",
            },
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="node-2",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-2")},
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="node-3",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-3")},
            metadata={
                "director": "Christopher Nolan",
            },
            embedding=[0.0, 0.0, 1.0],
        ),
    ]


@pytest.fixture(scope="session")
def node_embeddings_with_description() -> list[TextNode]:
    """Nodes with distinct text content for full text ranking tests."""
    return [
        TextNode(
            text="lorem ipsum dolor sit amet",
            id_="fts-node-1",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="fts-doc-1")},
            metadata={"author": "Stephen King"},
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="the quick brown fox jumps",
            id_="fts-node-2",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="fts-doc-2")},
            metadata={"author": "George Orwell"},
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="adipiscing elit sed consectetur",
            id_="fts-node-3",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="fts-doc-3")},
            metadata={"author": "J.K. Rowling"},
            embedding=[0.0, 0.0, 1.0],
        ),
    ]


@pytest.mark.skipif(not cosmosnosql_available, reason="cosmos client is not available")
class TestAzureCosmosNoSqlVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        # Purge any stale items left by a previous test session
        for container in (test_container, full_text_container):
            for item in container.query_items(
                query="SELECT * FROM c", enable_cross_partition_query=True
            ):
                container.delete_item(item, partition_key=item["id"])

    @classmethod
    def teardown_class(cls) -> None:
        # delete all the items in the containers
        for container in (test_container, full_text_container):
            for item in container.query_items(
                query="SELECT * FROM c", enable_cross_partition_query=True
            ):
                container.delete_item(item, partition_key=item["id"])

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the items in the containers
        for container in (test_container, full_text_container):
            for item in container.query_items(
                query="SELECT * FROM c", enable_cross_partition_query=True
            ):
                container.delete_item(item, partition_key=item["id"])

    def test_add_and_delete(self) -> None:
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            database_name=database_name,
            container_name=container_name,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
        )
        node = TextNode(
            text="test node text",
            id_="test-node-id",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-doc-id")
            },
            embedding=[0.5, 0.5, 0.5],
        )

        # --- add ---
        added_ids = vector_store.add([node])
        sleep(1)

        assert added_ids == ["test-node-id"], "add() must return the node id"

        items = list(test_container.read_all_items())
        assert len(items) == 1, "exactly one item should be stored after add"
        stored = items[0]
        assert stored["id"] == "test-node-id"
        assert stored["text"] == "test node text"
        assert "embedding" in stored, "embedding must be persisted"
        assert stored["embedding"] == [0.5, 0.5, 0.5]

        # --- delete ---
        vector_store.delete("test-node-id")

        items = list(test_container.read_all_items())
        assert len(items) == 0, "container must be empty after delete"

    def test_query(self, node_embeddings: List[TextNode]) -> None:
        """Default query (no search_type) returns top-1 closest node."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
            database_name=database_name,
            container_name=container_name,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )

        # Result shape
        assert res.nodes is not None
        assert res.ids is not None
        assert res.similarities is not None
        assert len(res.nodes) == 1
        assert len(res.ids) == 1
        assert len(res.similarities) == 1

        # Top result is the node with embedding [1,0,0] — exact cosine match
        assert res.nodes[0].get_content() == "lorem ipsum"
        assert res.ids[0] == "node-1"

        # CosmosDB cosine VectorDistance returns 1.0 for identical vectors (cosine similarity)
        assert res.similarities[0] == pytest.approx(1.0, abs=1e-5)

        # Node metadata is correctly restored
        assert res.nodes[0].metadata.get("author") == "Stephen King"
        assert res.nodes[0].metadata.get("theme") == "Friendship"

    def test_query_vector(self, node_embeddings: List[TextNode]) -> None:
        """Explicit search_type='vector' returns the same result as default query."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
            database_name=database_name,
            container_name=container_name,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1),
            search_type="vector",
        )

        assert len(res.nodes) == 1
        assert len(res.ids) == 1
        assert len(res.similarities) == 1

        # Closest node to [1,0,0] is the first fixture node
        assert res.ids[0] == "node-1"
        assert res.nodes[0].get_content() == "lorem ipsum"
        assert res.similarities[0] == pytest.approx(1.0, abs=1e-5)
        assert res.nodes[0].metadata.get("author") == "Stephen King"

    def test_query_vector_with_k(self, node_embeddings: List[TextNode]) -> None:
        """Vector search with k=2 returns exactly 2 results ranked by similarity."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
            database_name=database_name,
            container_name=container_name,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=2),
            search_type="vector",
        )

        assert len(res.nodes) == 2
        assert len(res.ids) == 2
        assert len(res.similarities) == 2

        # First result: exact match [1,0,0] — cosine similarity 1.0
        assert res.ids[0] == "node-1"
        assert res.nodes[0].get_content() == "lorem ipsum"
        assert res.similarities[0] == pytest.approx(1.0, abs=1e-3)

        # Second result must be a different node (less similar)
        assert res.ids[1] != res.ids[0]
        # Scores are ordered: first >= second (cosine similarity, higher = more similar)
        assert res.similarities[0] >= res.similarities[1]

    def test_query_vector_with_projection_mapping(
        self, node_embeddings: List[TextNode]
    ) -> None:
        """Projection mapping returns only requested fields as node metadata."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
            database_name=database_name,
            container_name=container_name,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1),
            search_type="vector",
            projection_mapping={"id": "id", "text": "text"},
        )

        assert len(res.nodes) == 1

        node = res.nodes[0]
        # Both projected fields are in metadata
        assert "id" in node.metadata
        assert "text" in node.metadata
        # Correct values from the closest node
        assert node.metadata["id"] == "node-1"
        assert node.metadata["text"] == "lorem ipsum"
        # Non-projected fields (e.g. author) must NOT appear in metadata
        assert "author" not in node.metadata
        assert "embedding" not in node.metadata

    def test_query_vector_with_offset_limit(
        self, node_embeddings: List[TextNode]
    ) -> None:
        """OFFSET 1 LIMIT 2 skips the top result and returns the next 2."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
            database_name=database_name,
            container_name=container_name,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)

        # Baseline: top-3 without offset to know expected order
        baseline = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3),
            search_type="vector",
        )

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3),
            search_type="vector",
            offset_limit="OFFSET 1 LIMIT 2",
        )

        assert len(res.nodes) == 2
        assert len(res.ids) == 2
        assert len(res.similarities) == 2

        # The offset result must NOT contain the top-1 node from baseline
        assert res.ids[0] != baseline.ids[0], "OFFSET 1 must skip the best match"
        # The offset result must match positions 1 and 2 from baseline
        assert res.ids[0] == baseline.ids[1]
        assert res.ids[1] == baseline.ids[2]

    def test_query_vector_with_where_filter(
        self, node_embeddings: List[TextNode]
    ) -> None:
        """WHERE filter restricts results to nodes matching the predicate."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
            database_name=database_name,
            container_name=container_name,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3),
            search_type="vector",
            where="c.metadata.author = 'Stephen King'",
        )

        # Only one node has author == "Stephen King"
        assert len(res.nodes) == 1
        assert len(res.ids) == 1

        node = res.nodes[0]
        assert res.ids[0] == "node-1"
        assert node.metadata.get("author") == "Stephen King"
        assert node.metadata.get("theme") == "Friendship"
        assert node.get_content() == "lorem ipsum"

        # No nodes from other authors should be present
        assert all(
            n.metadata.get("author") == "Stephen King" for n in res.nodes
        )

    def test_query_full_text_search(self, node_embeddings: List[TextNode]) -> None:
        """FullTextContains WHERE returns only nodes whose text contains the term."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=full_text_indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_full_text,
            database_name=database_name,
            container_name="full_text_container",
            full_text_search_enabled=True,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3),
            search_type="full_text_search",
            where="FullTextContains(c.text, 'lorem')",
        )

        # All 3 nodes contain "lorem ipsum" so all 3 should be returned
        assert len(res.nodes) == 3
        assert len(res.ids) == 3

        # Every returned node must contain "lorem" in its text
        for node in res.nodes:
            assert "lorem" in node.get_content(), (
                f"node '{node.node_id}' text '{node.get_content()}' must contain 'lorem'"
            )

        # All known IDs must be present (order is undefined for FTS without ranking)
        returned_ids = set(res.ids)
        assert "node-1" in returned_ids
        assert "node-2" in returned_ids
        assert "node-3" in returned_ids

        # FTS does not produce similarity scores — similarities list should be all 0.0
        assert all(s == 0.0 for s in res.similarities)

    def test_query_full_text_ranking(
        self, node_embeddings_with_description: List[TextNode]
    ) -> None:
        """Full text ranking orders nodes by FullTextScore descending."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=full_text_indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_full_text,
            database_name=database_name,
            container_name="full_text_container",
            full_text_search_enabled=True,
        )
        vector_store.add(node_embeddings_with_description)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3),
            search_type="full_text_ranking",
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "lorem ipsum"},
            ],
        )

        assert len(res.nodes) >= 1
        assert len(res.ids) == len(res.nodes)

        # Top result must contain "lorem ipsum" (only node d0 has both terms)
        top_content = res.nodes[0].get_content()
        assert "lorem" in top_content, f"top result '{top_content}' must contain 'lorem'"
        assert "ipsum" in top_content, f"top result '{top_content}' must contain 'ipsum'"

        # Top node ID must be the "lorem ipsum dolor sit amet" node
        assert res.ids[0] == "fts-node-1"

        # Top node metadata must be intact
        assert res.nodes[0].metadata.get("author") == "Stephen King"

        # No duplicates in returned IDs
        assert len(res.ids) == len(set(res.ids))

    def test_query_full_text_ranking_multiple_fields(
        self, node_embeddings_with_description: List[TextNode]
    ) -> None:
        """RRF fusion of two FullTextScore components ranks multi-term matches highest."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=full_text_indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_full_text,
            database_name=database_name,
            container_name="full_text_container",
            full_text_search_enabled=True,
        )
        vector_store.add(node_embeddings_with_description)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3),
            search_type="full_text_ranking",
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "lorem ipsum"},
                {"search_field": "text", "search_text": "dolor sit"},
            ],
        )

        assert len(res.nodes) >= 1
        assert len(res.ids) == len(res.nodes)

        # Node fts-node-1 ("lorem ipsum dolor sit amet") matches all four terms across both
        # RRF components — it must rank first
        assert res.ids[0] == "fts-node-1"
        top_content = res.nodes[0].get_content()
        assert "lorem" in top_content
        assert "dolor" in top_content

        # Other nodes that matched zero terms should appear after fts-node-1
        non_top_ids = res.ids[1:]
        assert "fts-node-1" not in non_top_ids

        # No duplicates
        assert len(res.ids) == len(set(res.ids))

    def test_query_hybrid_with_score_threshold(
        self, node_embeddings_with_description: List[TextNode]
    ) -> None:
        """Hybrid RRF (FullTextScore + VectorDistance) returns ranked results."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=full_text_indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_full_text,
            database_name=database_name,
            container_name="full_text_container",
            full_text_search_enabled=True,
        )
        vector_store.add(node_embeddings_with_description)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3),
            search_type="hybrid_score_threshold",
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "lorem ipsum"},
            ],
            threshold=0.0,
        )

        assert len(res.nodes) >= 1
        assert len(res.ids) == len(res.nodes)
        assert len(res.similarities) == len(res.nodes)

        # Node fts-node-1 scores highest: it matches "lorem ipsum" AND has the closest
        # vector [1,0,0] to the query [1,0,0]
        assert res.ids[0] == "fts-node-1"
        assert "lorem" in res.nodes[0].get_content()
        assert res.nodes[0].metadata.get("author") == "Stephen King"

        # All returned IDs are unique
        assert len(res.ids) == len(set(res.ids))

        # All returned nodes are TextNode instances with non-empty node_id
        for node in res.nodes:
            assert isinstance(node, TextNode)
            assert node.node_id

    def test_query_hybrid_with_weights(
        self, node_embeddings_with_description: List[TextNode]
    ) -> None:
        """Hybrid RRF returns nodes ordered by combined text + vector relevance."""
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=full_text_indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_full_text,
            database_name=database_name,
            container_name="full_text_container",
            full_text_search_enabled=True,
        )
        vector_store.add(node_embeddings_with_description)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3),
            search_type="hybrid",
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "lorem ipsum"},
            ],
        )

        assert len(res.nodes) >= 1
        assert len(res.ids) == len(res.nodes)
        assert len(res.similarities) == len(res.nodes)

        # Node fts-node-1 must rank first: best vector match [1,0,0] + best text match
        assert res.ids[0] == "fts-node-1"
        assert "lorem" in res.nodes[0].get_content()
        assert res.nodes[0].metadata.get("author") == "Stephen King"

        # All returned nodes are valid TextNode instances
        for node in res.nodes:
            assert isinstance(node, TextNode)
            assert node.node_id
            assert node.get_content()  # text must not be empty

        # No duplicate IDs
        assert len(res.ids) == len(set(res.ids))

    def test_query_weighted_hybrid_search(
        self, node_embeddings_with_description: List[TextNode]
    ) -> None:
        """Weighted hybrid RRF assigns per-component weights via weight=[...].

        weights=[0.3, 0.7] gives 30 % to FullTextScore and 70 % to VectorDistance.
        Node fts-node-1 has the exact query vector AND contains the search terms,
        so it must still rank first regardless of weight direction.
        """
        vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=test_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=full_text_indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_full_text,
            database_name=database_name,
            container_name="full_text_container",
            full_text_search_enabled=True,
        )
        vector_store.add(node_embeddings_with_description)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=3),
            search_type="weighted_hybrid_search",
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "lorem ipsum"},
            ],
            # 30 % text, 70 % vector — vector component gets higher weight
            weights=[0.3, 0.7],
        )

        assert len(res.nodes) >= 1
        assert len(res.ids) == len(res.nodes)
        assert len(res.similarities) == len(res.nodes)

        # fts-node-1 must still rank first: perfect vector match [1,0,0] + text match
        assert res.ids[0] == "fts-node-1"
        assert "lorem" in res.nodes[0].get_content()
        assert res.nodes[0].metadata.get("author") == "Stephen King"

        # All returned nodes are valid TextNode instances with non-empty content
        for node in res.nodes:
            assert isinstance(node, TextNode)
            assert node.node_id
            assert node.get_content()

        # No duplicate IDs
        assert len(res.ids) == len(set(res.ids))

    def test_cosmos_client_with_host_and_key(
        self, node_embeddings: List[TextNode]
    ) -> None:
        """from_host_and_key factory produces a working store."""
        vector_store = AzureCosmosDBNoSqlVectorSearch.from_host_and_key(
            host=URI,
            key=KEY,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
            database_name=database_name,
            container_name=container_name,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )

        assert len(res.nodes) == 1
        assert len(res.ids) == 1
        assert len(res.similarities) == 1

        assert res.ids[0] == "node-1"
        assert res.nodes[0].get_content() == "lorem ipsum"
        assert res.similarities[0] == pytest.approx(1.0, abs=1e-5)
        assert res.nodes[0].metadata.get("author") == "Stephen King"

    def test_cosmos_client_with_connection_string(
        self, node_embeddings: List[TextNode]
    ) -> None:
        """from_connection_string factory produces a working store."""
        connection_string = "AccountEndpoint=" + URI + ";" + "AccountKey=" + KEY + ";"
        vector_store = AzureCosmosDBNoSqlVectorSearch.from_connection_string(
            connection_string=connection_string,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_database_properties=cosmos_database_properties_test,
            cosmos_container_properties=cosmos_container_properties_test,
            database_name=database_name,
            container_name=container_name,
        )
        vector_store.add(node_embeddings)  # type: ignore
        sleep(1)

        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )

        assert len(res.nodes) == 1
        assert len(res.ids) == 1
        assert len(res.similarities) == 1

        assert res.ids[0] == "node-1"
        assert res.nodes[0].get_content() == "lorem ipsum"
        assert res.similarities[0] == pytest.approx(1.0, abs=1e-5)
        assert res.nodes[0].metadata.get("author") == "Stephen King"
