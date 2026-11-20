"""Cassandra Document Store implementation for LlamaIndex."""

from typing import Optional

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE
from llama_index.storage.kvstore.cassandra import CassandraKVStore


class CassandraDocumentStore(KVDocumentStore):
    """
    Cassandra Document (Node) store.

    A Cassandra store for Document and Node objects.

    Args:
        cassandra_kvstore (CassandraKVStore): Cassandra key-value store
        namespace (str): namespace for the docstore

    """

    def __init__(
        self,
        cassandra_kvstore: CassandraKVStore,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Init a CassandraDocumentStore."""
        super().__init__(
            cassandra_kvstore,
            namespace=namespace,
            batch_size=batch_size,
            node_collection_suffix=node_collection_suffix,
            ref_doc_collection_suffix=ref_doc_collection_suffix,
            metadata_collection_suffix=metadata_collection_suffix,
        )

    @classmethod
    def from_params(
        cls,
        contact_points: Optional[list] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        keyspace: Optional[str] = None,
        table: Optional[str] = None,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> "CassandraDocumentStore":
        """Load a CassandraDocumentStore from connection parameters."""
        cassandra_kvstore = CassandraKVStore.from_params(
            contact_points=contact_points,
            port=port,
            username=username,
            password=password,
            keyspace=keyspace,
            table=table,
        )
        return cls(
            cassandra_kvstore,
            namespace=namespace,
            node_collection_suffix=node_collection_suffix,
            ref_doc_collection_suffix=ref_doc_collection_suffix,
            metadata_collection_suffix=metadata_collection_suffix,
            batch_size=batch_size,
        )

    @classmethod
    def from_kvstore(
        cls,
        kvstore: CassandraKVStore,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> "CassandraDocumentStore":
        """Create a CassandraDocumentStore from an existing CassandraKVStore."""
        return cls(
            kvstore,
            namespace=namespace,
            node_collection_suffix=node_collection_suffix,
            ref_doc_collection_suffix=ref_doc_collection_suffix,
            metadata_collection_suffix=metadata_collection_suffix,
            batch_size=batch_size,
        )
