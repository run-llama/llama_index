from typing import Any
from llama_index.core.storage.kvstore.types import BaseKVStore
import pytest
from llama_index.storage.kvstore.couchbase import CouchbaseKVStore
import os
from datetime import timedelta

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions

CONNECTION_STRING = os.getenv("COUCHBASE_CONNECTION_STRING", "")
BUCKET_NAME = os.getenv("COUCHBASE_BUCKET_NAME", "")
SCOPE_NAME = os.getenv("COUCHBASE_SCOPE_NAME", "")
COLLECTION_NAME = os.getenv("COUCHBASE_COLLECTION_NAME", "")
USERNAME = os.getenv("COUCHBASE_USERNAME", "")
PASSWORD = os.getenv("COUCHBASE_PASSWORD", "")

SLEEP_DURATION = 1


def test_class():
    names_of_base_classes = [b.__name__ for b in CouchbaseKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


## TestCouchbaseKVStore
def set_all_env_vars() -> bool:
    """Check if all required environment variables are set."""
    return all(
        [
            CONNECTION_STRING,
            BUCKET_NAME,
            SCOPE_NAME,
            COLLECTION_NAME,
            USERNAME,
            PASSWORD,
        ]
    )


def get_cluster() -> Any:
    """Get a couchbase cluster object."""
    auth = PasswordAuthenticator(USERNAME, PASSWORD)
    options = ClusterOptions(auth)
    connect_string = CONNECTION_STRING
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster  # , acluster


@pytest.fixture()
def cluster() -> Any:
    """Get a couchbase cluster object."""
    return get_cluster()


@pytest.mark.skipif(
    not set_all_env_vars(),
    reason="Required environment variables for Couchbase not set.",
)
class TestCouchbaseKVStore:
    @classmethod
    def setup_class(self) -> None:
        self.cluster = get_cluster()
        self.kvstore = CouchbaseKVStore.from_couchbase_client(
            self.cluster,
            BUCKET_NAME,
            SCOPE_NAME,
        )

    def test_add_key_value_pair(self):
        """Test adding a key-value pair to the store."""
        key = "key1"
        value = {"doc": "value1", "status": "active"}
        self.kvstore.put(key, value)
        doc = self.kvstore.get(key)
        assert doc == value

    def test_add_key_value_pairs(self):
        """Test adding multiple key-value pairs to the store."""
        key1 = "key1"
        value1 = {"doc": "value1", "status": "active"}
        key2 = "key2"
        value2 = {"doc": "value2", "status": "inactive"}

        key_value_pairs = [
            (key1, value1),
            (key2, value2),
        ]

        self.kvstore.put_all(key_value_pairs)

        doc1 = self.kvstore.get(key1)
        doc2 = self.kvstore.get(key2)

        assert doc1 == value1
        assert doc2 == value2

    def test_delete_key_value_pair(self):
        """Test deleting a key-value pair from the store."""
        key = "key1"
        value = {"doc": "value1", "status": "active"}
        self.kvstore.put(key, value)
        doc = self.kvstore.get(key)
        assert doc == value

        is_deleted = self.kvstore.delete(key)
        assert is_deleted

        doc = self.kvstore.get(key)
        assert doc is None

    def test_get_all_key_value_pairs(self):
        """Test getting all key-value pairs from the store."""
        key1 = "key1"
        value1 = {"doc": "value1", "status": "active"}
        key2 = "key2"
        value2 = {"doc": "value2", "status": "inactive"}

        key_value_pairs = [
            (key1, value1),
            (key2, value2),
        ]

        self.kvstore.put_all(key_value_pairs, batch_size=2)

        docs = self.kvstore.get_all()
        assert len(docs) == 2
        assert key1 in docs
        assert key2 in docs

    def test_delete_multiple_key_value_pairs(self):
        """Test deleting multiple key-value pairs from the store."""
        key1 = "key1"
        value1 = {"doc": "value1", "status": "active"}
        key2 = "key2"
        value2 = {"doc": "value2", "status": "inactive"}

        key_value_pairs = [
            (key1, value1),
            (key2, value2),
        ]

        self.kvstore.put_all(key_value_pairs, batch_size=2)

        docs = self.kvstore.get_all()
        assert len(docs) == 2

        is_deleted = self.kvstore.delete(key1)
        assert is_deleted

        doc1 = self.kvstore.get(key1)
        assert doc1 is None

        is_deleted = self.kvstore.delete(key2)
        assert is_deleted

        doc2 = self.kvstore.get(key2)
        assert doc2 is None

    def test_non_default_collection(self):
        """Test adding a key-value pair to a non-default collection."""
        key = "key1"
        value = {"doc": "value1", "status": "active"}

        collection = "test_collection"
        self.kvstore.put(key, value, collection=collection)
        doc = self.kvstore.get(key, collection=collection)

        assert doc == value

        is_deleted = self.kvstore.delete(key, collection=collection)
        assert is_deleted
        doc = self.kvstore.get(key, collection=collection)
        assert doc is None
