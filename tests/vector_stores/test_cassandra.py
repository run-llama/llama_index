import sys
import unittest
from unittest.mock import MagicMock

import pytest
from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.cassandra import CassandraVectorStore
from llama_index.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

try:
    import cassio

    has_cassio = True
except ImportError:
    has_cassio = False


class TestCassandraVectorStore(unittest.TestCase):
    @pytest.mark.skipif(not has_cassio, reason="cassio not installed")
    def test_cassandra_create_and_crud(self) -> None:
        mock_db_session = MagicMock()
        try:
            import cassio
        except ModuleNotFoundError:
            # mock `cassio` if not installed
            mock_cassio = MagicMock()
            sys.modules["cassio"] = mock_cassio
        #
        vector_store = CassandraVectorStore(
            session=mock_db_session,
            keyspace="keyspace",
            table="table",
            embedding_dimension=2,
            ttl_seconds=123,
        )

        vector_store.add(
            [
                TextNode(
                    text="test node text",
                    id_="test node id",
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc id")
                    },
                    embedding=[0.5, 0.5],
                )
            ]
        )

        vector_store.delete("test node id")

        vector_store.client

    @pytest.mark.skipif(not has_cassio, reason="cassio not installed")
    def test_cassandra_queries(self) -> None:
        mock_db_session = MagicMock()
        try:
            import cassio
        except ModuleNotFoundError:
            # mock `cassio` if not installed
            mock_cassio = MagicMock()
            sys.modules["cassio"] = mock_cassio
        #
        vector_store = CassandraVectorStore(
            session=mock_db_session,
            keyspace="keyspace",
            table="table",
            embedding_dimension=2,
            ttl_seconds=123,
        )
        # q1: default
        query = VectorStoreQuery(
            query_embedding=[1, 1],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        vector_store.query(
            query,
        )
        # q2: mmr, threshold in query takes precedence
        query = VectorStoreQuery(
            query_embedding=[1, 1],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
            mmr_threshold=0.45,
        )
        vector_store.query(
            query,
            mmr_threshold=0.9,
        )
        # q3: mmr, threshold defined as param to `query`
        query = VectorStoreQuery(
            query_embedding=[1, 1],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        vector_store.query(
            query,
            mmr_threshold=0.9,
        )
        # q4: mmr, prefetch control
        query = VectorStoreQuery(
            query_embedding=[1, 1],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        vector_store.query(
            query,
            mmr_prefetch_factor=7.7,
        )
        # q5: mmr, conflicting prefetch control directives
        query = VectorStoreQuery(
            query_embedding=[1, 1],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        with pytest.raises(ValueError):
            vector_store.query(
                query,
                mmr_prefetch_factor=7.7,
                mmr_prefetch_k=80,
            )
