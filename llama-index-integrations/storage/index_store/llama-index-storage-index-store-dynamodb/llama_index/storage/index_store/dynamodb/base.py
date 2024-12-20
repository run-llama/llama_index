from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.dynamodb import DynamoDBKVStore


class DynamoDBIndexStore(KVIndexStore):
    def __init__(
        self,
        dynamodb_kvstore: DynamoDBKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a DynamoDBIndexStore."""
        super().__init__(
            kvstore=dynamodb_kvstore,
            namespace=namespace,
            collection_suffix=collection_suffix,
        )

    @classmethod
    def from_table_name(
        cls,
        table_name: str,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> "DynamoDBIndexStore":
        """Load DynamoDBIndexStore from a DynamoDB table name."""
        ddb_kvstore = DynamoDBKVStore.from_table_name(table_name=table_name)
        return cls(
            dynamodb_kvstore=ddb_kvstore,
            namespace=namespace,
            collection_suffix=collection_suffix,
        )
