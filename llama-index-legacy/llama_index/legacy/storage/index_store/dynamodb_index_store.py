from __future__ import annotations

from llama_index.legacy.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.legacy.storage.kvstore.dynamodb_kvstore import DynamoDBKVStore


class DynamoDBIndexStore(KVIndexStore):
    def __init__(self, dynamodb_kvstore: DynamoDBKVStore, namespace: str | None = None):
        """Init a DynamoDBIndexStore."""
        super().__init__(kvstore=dynamodb_kvstore, namespace=namespace)

    @classmethod
    def from_table_name(
        cls, table_name: str, namespace: str | None = None
    ) -> DynamoDBIndexStore:
        """Load DynamoDBIndexStore from a DynamoDB table name."""
        ddb_kvstore = DynamoDBKVStore.from_table_name(table_name=table_name)
        return cls(dynamodb_kvstore=ddb_kvstore, namespace=namespace)
