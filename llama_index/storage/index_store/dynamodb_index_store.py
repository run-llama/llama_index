from __future__ import annotations

from typing import Optional

from llama_index.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.dynamodb_kvstore import DynamoDBKVStore


class DynamoDBIndexStore(KVIndexStore):
    def __init__(
        self, dynamodb_kvstore: DynamoDBKVStore, namespace: Optional[str] = None
    ):
        """Init a DynamoDBIndexStore."""
        super().__init__(kvstore=dynamodb_kvstore, namespace=namespace)

    @classmethod
    def from_table_name(
        cls, table_name: str, namespace: Optional[str] = None
    ) -> DynamoDBIndexStore:
        """Load DynamoDBIndexStore from a DynamoDB table name."""
        ddb_kvstore = DynamoDBKVStore.from_table_name(table_name=table_name)
        return cls(dynamodb_kvstore=ddb_kvstore, namespace=namespace)
