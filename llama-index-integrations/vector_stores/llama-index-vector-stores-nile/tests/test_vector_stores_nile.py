import unittest
import pytest
from unittest import mock

import numpy as np

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.nile import NileVectorStore


class TestNileVectorStore(unittest.TestCase):
    @pytest.fixture(autouse=True)
    @mock.patch("psycopg.connect")
    def vector_store_setup(self, mock_connect):
        # Mock the psycopg connection and cursor
        self.mock_connection = (
            mock_connect.return_value
        )  # result of psycopg2.connect(**connection_stuff)
        self.mock_cursor = (
            self.mock_connection.cursor.return_value
        )  # result of con.cursor(cursor_factory=DictCursor)
        self.vector_store = NileVectorStore(
            service_url="postgresql://user:password@localhost/dbname",
            table_name="test_table",
        )
        self.tenant_aware_vector_store = NileVectorStore(
            service_url="postgresql://user:password@localhost/dbname",
            table_name="test_table",
            tenant_aware=True,
        )

    def test_class(self):
        names_of_base_classes = [b.__name__ for b in NileVectorStore.__mro__]
        self.assertIn(BasePydanticVectorStore.__name__, names_of_base_classes)

    def test_add(self):
        node = TextNode(
            text="Test node", embedding=np.array([0.1, 0.2, 0.3]), id_="test_id"
        )
        self.vector_store.add([node])
        # create table twice when initializing the fixture, and insert one row
        assert self.mock_connection.commit.call_count == 3

        # expected to fail if we don't have tenant_id in the node metadata
        with pytest.raises(Exception) as e_info:
            self.tenant_aware_vector_store.add([node])
            assert "tenant_id cannot be None if tenant_aware is True" in str(
                e_info.value
            )

        # one more insert for the new node
        node.metadata["tenant_id"] = "test_tenant"
        self.tenant_aware_vector_store.add([node])
        assert self.mock_connection.commit.call_count == 4

    def test_delete(self):
        self.vector_store.delete("test_id")
        # create table twice when initializing the fixture, and delete one row
        assert self.mock_connection.commit.call_count == 3

        # expected to fail if we don't have tenant_id in kwargs
        with pytest.raises(Exception) as e_info:
            self.tenant_aware_vector_store.delete("test_id")
            assert (
                "tenant_id must be specified in delete_kwargs if tenant_aware is True"
                in str(e_info.value)
            )

        # delete the node with tenant_id
        self.tenant_aware_vector_store.delete("test_id", tenant_id="test_tenant")
        assert self.mock_connection.commit.call_count == 4

    def test_query(self):
        query_embedding = VectorStoreQuery(
            query_embedding=np.array([0.1, 0.2, 0.3]), similarity_top_k=2
        )
        results = self.vector_store.query(query_embedding)
        assert isinstance(results, VectorStoreQueryResult)

        # expected to fail if we don't have tenant_id in kwargs
        with pytest.raises(Exception) as e_info:
            self.tenant_aware_vector_store.query(query_embedding)
            assert (
                "tenant_id must be specified in query_kwargs if tenant_aware is True"
                in str(e_info.value)
            )

        # query the node with tenant_id
        results = self.tenant_aware_vector_store.query(
            query_embedding, tenant_id="test_tenant"
        )
        assert isinstance(results, VectorStoreQueryResult)


if __name__ == "__main__":
    unittest.main()
