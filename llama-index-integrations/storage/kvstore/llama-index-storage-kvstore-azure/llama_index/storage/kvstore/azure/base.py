from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)
from llama_index.utils.azure import (
    ServiceMode,
    deserialize,
    sanitize_table_name,
    serialize,
)

IMPORT_ERROR_MSG = (
    "`azure-data-tables` package not found, please run `pip install azure-data-tables`"
)
MISSING_ASYNC_CLIENT_ERROR_MSG = "AzureKVStore was not initialized with an async client"
DEFAULT_PARTITION_KEY = "default"


class AzureKVStore(BaseKVStore):
    """
    Provides a key-value store interface for Azure Table Storage and Cosmos
    DB. This class supports both synchronous and asynchronous operations on
    Azure Table Storage and Cosmos DB. It supports connecting to the service
    using different credentials and manages table creation and data
    serialization to conform to the storage requirements.
    """

    partition_key: str

    def __init__(
        self,
        table_service_client: Any,
        atable_service_client: Optional[Any] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the AzureKVStore with Azure Table Storage clients."""
        try:
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self.service_mode = service_mode
        self.partition_key = (
            DEFAULT_PARTITION_KEY if partition_key is None else partition_key
        )

        super().__init__(*args, **kwargs)

        self._table_service_client = cast(TableServiceClient, table_service_client)
        self._atable_service_client = cast(
            Optional[AsyncTableServiceClient], atable_service_client
        )

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """Creates an instance of AzureKVStore using a connection string."""
        try:
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_service_client = TableServiceClient.from_connection_string(
            connection_string
        )
        atable_service_client = AsyncTableServiceClient.from_connection_string(
            connection_string
        )
        return cls(
            table_service_client,
            atable_service_client,
            service_mode,
            partition_key,
            *args,
            **kwargs,
        )

    @classmethod
    def from_account_and_key(
        cls,
        account_name: str,
        account_key: str,
        endpoint: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """Creates an instance of AzureKVStore from an account name and key."""
        try:
            from azure.core.credentials import AzureNamedKeyCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if endpoint is None:
            endpoint = f"https://{account_name}.table.core.windows.net"
        credential = AzureNamedKeyCredential(account_name, account_key)
        return cls._from_clients(
            endpoint, credential, service_mode, partition_key, *args, **kwargs
        )

    @classmethod
    def from_account_and_id(
        cls,
        account_name: str,
        endpoint: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """Creates an instance of AzureKVStore from an account name and managed ID."""
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if endpoint is None:
            endpoint = f"https://{account_name}.table.core.windows.net"
        credential = DefaultAzureCredential()
        return cls._from_clients(
            endpoint, credential, service_mode, partition_key, *args, **kwargs
        )

    @classmethod
    def from_sas_token(
        cls,
        endpoint: str,
        sas_token: str,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """Creates an instance of AzureKVStore using a SAS token."""
        try:
            from azure.core.credentials import AzureSasCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = AzureSasCredential(sas_token)
        return cls._from_clients(
            endpoint, credential, service_mode, partition_key, *args, **kwargs
        )

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """
        Creates an instance of AzureKVStore using Azure Active Directory
        (AAD) tokens.
        """
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = DefaultAzureCredential()
        return cls._from_clients(
            endpoint, credential, service_mode, partition_key, *args, **kwargs
        )

    def put(
        self,
        key: str,
        val: dict,
        collection: str = None,
    ) -> None:
        """Inserts or replaces a key-value pair in the specified table."""
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )
        table_client = self._table_service_client.create_table_if_not_exists(table_name)
        table_client.upsert_entity(
            {
                "PartitionKey": self.partition_key,
                "RowKey": key,
                **serialize(self.service_mode, val),
            },
            UpdateMode.REPLACE,
        )

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = None,
    ) -> None:
        """Inserts or replaces a key-value pair in the specified table."""
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )
        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )
        await atable_client.upsert_entity(
            {
                "PartitionKey": self.partition_key,
                "RowKey": key,
                **serialize(self.service_mode, val),
            },
            mode=UpdateMode.REPLACE,
        )

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Inserts or replaces multiple key-value pairs in the specified table.
        """
        try:
            from azure.data.tables import TransactionOperation
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )
        table_client = self._table_service_client.create_table_if_not_exists(table_name)

        entities = [
            {
                "PartitionKey": self.partition_key,
                "RowKey": key,
                **serialize(self.service_mode, val),
            }
            for key, val in kv_pairs
        ]

        entities_len = len(entities)
        for start in range(0, entities_len, batch_size):
            table_client.submit_transaction(
                (TransactionOperation.UPSERT, entities[i])
                for i in range(start, min(start + batch_size, entities_len))
            )

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Inserts or replaces multiple key-value pairs in the specified table.
        """
        try:
            from azure.data.tables import TransactionOperation
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )

        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )

        entities = [
            {
                "PartitionKey": self.partition_key,
                "RowKey": key,
                **serialize(self.service_mode, val),
            }
            for key, val in kv_pairs
        ]

        entities_len = len(entities)
        for start in range(0, entities_len, batch_size):
            await atable_client.submit_transaction(
                (TransactionOperation.UPSERT, entities[i])
                for i in range(start, min(start + batch_size, entities_len))
            )

    def get(
        self,
        key: str,
        collection: str = None,
        select: Optional[Union[str, List[str]]] = None,
    ) -> Optional[dict]:
        """Retrieves a value by key from the specified table."""
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )

        table_client = self._table_service_client.create_table_if_not_exists(table_name)
        try:
            entity = table_client.get_entity(
                partition_key=self.partition_key, row_key=key, select=select
            )
            return deserialize(self.service_mode, entity)
        except ResourceNotFoundError:
            return None

    async def aget(
        self,
        key: str,
        collection: str = None,
        select: Optional[Union[str, List[str]]] = None,
    ) -> Optional[dict]:
        """Retrieves a value by key from the specified table."""
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )
        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )
        try:
            entity = await atable_client.get_entity(
                partition_key=self.partition_key, row_key=key, select=select
            )
            return deserialize(self.service_mode, entity)
        except ResourceNotFoundError:
            return None

    def get_all(
        self,
        collection: str = None,
        select: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, dict]:
        """
        Retrieves all key-value pairs from a specified partition in the table.
        """
        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )
        table_client = self._table_service_client.create_table_if_not_exists(table_name)
        entities = table_client.list_entities(
            filter=f"PartitionKey eq '{self.partition_key}'",
            select=select,
        )
        return {
            entity["RowKey"]: deserialize(self.service_mode, entity)
            for entity in entities
        }

    async def aget_all(
        self,
        collection: str = None,
        select: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, dict]:
        """
        Retrieves all key-value pairs from a specified partition in the table.
        """
        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)
        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )
        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )
        entities = atable_client.list_entities(
            filter=f"PartitionKey eq '{self.partition_key}'",
            select=select,
        )
        return {
            entity["RowKey"]: deserialize(self.service_mode, entity)
            async for entity in entities
        }

    def delete(
        self,
        key: str,
        collection: str = None,
    ) -> bool:
        """
        Deletes a specific key-value pair from the store based on the
        provided key and partition key.
        """
        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )
        table_client = self._table_service_client.create_table_if_not_exists(table_name)
        table_client.delete_entity(partition_key=self.partition_key, row_key=key)
        return True

    async def adelete(
        self,
        key: str,
        collection: str = None,
    ) -> bool:
        """Asynchronously deletes a specific key-value pair from the store."""
        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)
        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )
        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )
        await atable_client.delete_entity(partition_key=self.partition_key, row_key=key)
        return True

    def query(
        self,
        query_filter: str,
        collection: str = None,
        select: Optional[Union[str, List[str]]] = None,
    ) -> Generator[dict, None, None]:
        """Retrieves a value by key from the specified table."""
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )

        table_client = self._table_service_client.create_table_if_not_exists(table_name)
        try:
            entities = table_client.query_entities(
                query_filter=query_filter, select=select
            )

            return (deserialize(self.service_mode, entity) for entity in entities)
        except ResourceNotFoundError:
            return None

    async def aquery(
        self,
        query_filter: str,
        collection: str = None,
        select: Optional[Union[str, List[str]]] = None,
    ) -> Optional[AsyncGenerator[dict, None]]:
        """Asynchronously retrieves a value by key from the specified table."""
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION if not collection else sanitize_table_name(collection)
        )
        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )
        try:
            entities = atable_client.query_entities(
                query_filter=query_filter, select=select
            )

            return (deserialize(self.service_mode, entity) async for entity in entities)

        except ResourceNotFoundError:
            return None

    @classmethod
    def _from_clients(
        cls, endpoint: str, credential: Any, *args: Any, **kwargs: Any
    ) -> "AzureKVStore":
        """
        Private method to create synchronous and asynchronous table service
        clients.
        """
        try:
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_client = TableServiceClient(endpoint=endpoint, credential=credential)
        atable_client = AsyncTableServiceClient(
            endpoint=endpoint, credential=credential
        )
        return cls(table_client, atable_client, *args, **kwargs)
