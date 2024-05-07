from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

IMPORT_ERROR_MSG = (
    "`azure-data-tables` package not found, please run `pip install azure-data-tables`"
)


class AzureKVStore(BaseKVStore):
    """Azure Table Key-Value store.

    Args:
        table_client (Any): Azure Table client
        atable_client (Optional[Any]): Azure Table async client
        table_name (str): Azure Table name (defaults to 'table_docstore')

    """

    def __init__(
        self,
        table_client: Any,
        atable_client: Optional[Any] = None,
        table_name: str = "table_docstore",
    ) -> None:
        """Init an AzureKVStore."""
        try:
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._table_client = cast(TableServiceClient, table_client)
        self._atable_client = cast(Optional[AsyncTableServiceClient], atable_client)
        self._table_name = table_name

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        table_name: Optional[str] = None,
    ) -> "AzureKVStore":
        try:
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)
        table_name = table_name or "table_docstore"
        table_client = TableServiceClient.from_connection_string(connection_string)
        atable_client = AsyncTableServiceClient.from_connection_string(
            connection_string
        )
        return cls(table_client, atable_client, table_name)

    @classmethod
    def from_account_and_key(
        cls, account_name: str, account_key: str, table_name: Optional[str] = None
    ) -> "AzureKVStore":
        """Load an AzureKVStore from an account name and key.

        Args:
            account_name (str): Azure Storage Account Name
            account_key (str): Azure Storage Account Key
            table_name (Optional[str]): Azure Table name

        """
        try:
            from azure.core.credentials import AzureNamedKeyCredential
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = AzureNamedKeyCredential(account_name, account_key)
        endpoint = f"https://{account_name}.table.core.windows.net"
        table_client = TableServiceClient(endpoint=endpoint, credential=credential)
        atable_client = AsyncTableServiceClient(
            endpoint=endpoint, credential=credential
        )
        table_name = table_name or "table_docstore"
        return cls(table_client, atable_client, table_name)

    @classmethod
    def from_sas_token(
        cls, endpoint: str, sas_token: str, table_name: Optional[str] = None
    ) -> "AzureKVStore":
        """Load an AzureKVStore from a SAS token.

        Args:
            endpoint (str): Azure Table service endpoint
            sas_token (str): Shared Access Signature token
            table_name (Optional[str]): Azure Table name

        """
        try:
            from azure.core.credentials import AzureSasCredential
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = AzureSasCredential(sas_token)
        table_client = TableServiceClient(endpoint=endpoint, credential=credential)
        atable_client = AsyncTableServiceClient(
            endpoint=endpoint, credential=credential
        )
        table_name = table_name or "table_docstore"
        return cls(table_client, atable_client, table_name)

    @classmethod
    def from_aad_token(
        cls, endpoint: str, table_name: Optional[str] = None
    ) -> "AzureKVStore":
        """Load an AzureKVStore from an AAD token.

        Args:
            endpoint (str): Azure Table service endpoint
            table_name (Optional[str]): Azure Table name

        """
        try:
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = DefaultAzureCredential()
        table_client = TableServiceClient(endpoint=endpoint, credential=credential)
        atable_client = AsyncTableServiceClient(
            endpoint=endpoint, credential=credential
        )
        table_name = table_name or "table_docstore"
        return cls(table_client, atable_client, table_name)

    def _check_async_client(self) -> None:
        if self._adb is None:
            raise ValueError("AzureKVStore was not initialized with an async client")

    def put(
        self,
        key: str,
        val: dict,
        table: str = DEFAULT_COLLECTION,
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            table (str): table name

        """
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        entity = {"PartitionKey": "default", "RowKey": key, **val}
        table_client = self._table_client.get_table_client(table)
        table_client.upsert_entity(entity, mode=UpdateMode.REPLACE)

    async def aput(
        self,
        key: str,
        val: dict,
        table: str = DEFAULT_COLLECTION,
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): table name

        """
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._check_async_client()
        entity = {"PartitionKey": "default", "RowKey": key, **val}
        table_client = self._atable_client.get_table_client(table)
        await table_client.upsert_entity(entity, mode=UpdateMode.REPLACE)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        table: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Put multiple key-value pairs into the store.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            table (str): table name
            batch_size (int): batch size

        """
        entities = [
            {"PartitionKey": "default", "RowKey": key, **value}
            for key, value in kv_pairs
        ]
        table_client = self._table_client.get_table_client(table)
        for batch in (
            entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
        ):
            table_client.submit_transaction([{"upsert": entity} for entity in batch])

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        table: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Put multiple key-value pairs into the store asynchronously.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            table (str): table name
            batch_size (int): batch size

        """
        self._check_async_client()
        entities = [
            {"PartitionKey": "default", "RowKey": key, **value}
            for key, value in kv_pairs
        ]
        table_client = self._atable_client.get_table_client(table)
        for batch in (
            entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
        ):
            await table_client.submit_transaction(
                [{"upsert": entity} for entity in batch]
            )

    def get(self, key: str, table: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            table (str): table name

        """
        table_client = self._table_client.get_table_client(table)
        try:
            entity = table_client.get_entity(partition_key="default", row_key=key)
            entity.pop("PartitionKey", None)
            entity.pop("RowKey", None)
            return entity
        except KeyError:
            return None

    async def aget(self, key: str, table: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store asynchronously.

        Args:
            key (str): key
            table (str): table name

        """
        self._check_async_client()
        table_client = self._atable_client.get_table_client(table)
        try:
            entity = await table_client.get_entity(partition_key="default", row_key=key)
            entity.pop("PartitionKey", None)
            entity.pop("RowKey", None)
            return entity
        except KeyError:
            return None

    def get_all(self, table: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            table (str): table name

        """
        table_client = self._table_client.get_table_client(table)
        entities = table_client.list_entities(filter="PartitionKey eq 'default'")
        output = {}
        for entity in entities:
            key = entity.pop("RowKey")
            entity.pop("PartitionKey", None)
            output[key] = entity
        return output

    async def aget_all(self, table: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store asynchronously.

        Args:
            table (str): table name

        """
        self._check_async_client()
        table_client = self._atable_client.get_table_client(table)
        entities = table_client.list_entities(filter="PartitionKey eq 'default'")
        output = {}
        async for entity in entities:
            key = entity.pop("RowKey")
            entity.pop("PartitionKey", None)
            output[key] = entity
        return output

    def delete(self, key: str, table: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store. Always returns True.

        Args:
            key (str): key
            table (str): table name

        """
        table_client = self._table_client.get_table_client(table)
        table_client.delete_entity(partition_key="default", row_key=key)
        return True

    async def adelete(self, key: str, table: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store asynchronously. Always returns True.

        Args:
            key (str): key
            table (str): table name

        """
        self._check_async_client()
        table_client = self._atable_client.get_table_client(table)
        await table_client.delete_entity(partition_key="default", row_key=key)
        return True
