import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

IMPORT_ERROR_MSG = (
    "`azure-data-tables` package not found, please run `pip install azure-data-tables`"
)

# https://learn.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model#table-names
ALPHANUMERIC_REGEX = re.compile(r"[^A-Za-z0-9]")
VALID_TABLE_NAME_REGEX = re.compile(r"^[A-Za-z][A-Za-z0-9]{2,62}$")
DEFAULT_PARTITION_KEY = "default"


class AzureKVStore(BaseKVStore):
    """Azure Table Key-Value store.

    Args:
        table_client (Any): Azure Table client
        atable_client (Optional[Any]): Azure Table async client
        partition_key (Optional[str]): The partition key to use
    """

    def __init__(
        self,
        table_client: Any,
        atable_client: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Init an AzureKVStore."""
        super().__init__(*args, **kwargs)
        try:
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._table_client = cast(TableServiceClient, table_client)
        self._atable_client = cast(Optional[AsyncTableServiceClient], atable_client)

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """Load an AzureKVStore from a connection string to an Azure Storage or Cosmos account.

        Args:
            connection_string (str): A connection string to an Azure Storage or Cosmos account.
        """
        try:
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_client = TableServiceClient.from_connection_string(connection_string)
        atable_client = AsyncTableServiceClient.from_connection_string(
            connection_string
        )
        return cls(table_client, atable_client, *args, **kwargs)

    @classmethod
    def from_account_and_key(
        cls,
        account_name: str,
        account_key: str,
        endpoint: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """Load an AzureKVStore from an account name and key.

        Args:
            account_name (str): Azure Storage Account name
            account_key (str): Azure Storage Account key
            endpoint (Optional[str]): Azure Storage Account endpoint. Defaults to None.
        """
        try:
            from azure.core.credentials import AzureNamedKeyCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if endpoint is None:
            endpoint = f"https://{account_name}.table.core.windows.net"
        credential = AzureNamedKeyCredential(account_name, account_key)
        return cls._create_clients(endpoint, credential, *args, **kwargs)

    @classmethod
    def from_sas_token(
        cls, endpoint: str, sas_token: str, *args: Any, **kwargs: Any
    ) -> "AzureKVStore":
        """Load an AzureKVStore from a SAS token.

        Args:
            endpoint (str): Azure Table service endpoint
            sas_token (str): Shared Access Signature token
        """
        try:
            from azure.core.credentials import AzureSasCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = AzureSasCredential(sas_token)
        return cls._create_clients(endpoint, credential, *args, **kwargs)

    @classmethod
    def from_aad_token(cls, endpoint: str, *args: Any, **kwargs: Any) -> "AzureKVStore":
        """Load an AzureKVStore from an AAD token.

        Args:
            endpoint (str): Azure Table service endpoint
            table_name (Optional[str]): Azure Table name
        """
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = DefaultAzureCredential()
        return cls._create_clients(endpoint, credential, *args, **kwargs)

    def _check_async_client(self) -> None:
        if self._adb is None:
            raise ValueError("AzureKVStore was not initialized with an async client")

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
        partition_key: str = DEFAULT_PARTITION_KEY,
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

        table_name = self._sanitize_table_name(collection)
        table_client = self._table_client.create_table_if_not_exists(table_name)
        entity = {
            "PartitionKey": partition_key,
            "RowKey": key,
            **self._serialize(val),
        }
        table_client.upsert_entity(entity, mode=UpdateMode.REPLACE)

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
        partition_key: str = DEFAULT_PARTITION_KEY,
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

        table_name = self._sanitize_table_name(collection)
        self._check_async_client()
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        await table_client.upsert_entity(
            {
                "PartitionKey": partition_key,
                "RowKey": key,
                **self._serialize(val),
            },
            mode=UpdateMode.REPLACE,
        )

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> None:
        """Put multiple key-value pairs into the store.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            collection (str): table name
            batch_size (int): batch size
        """
        table_name = self._sanitize_table_name(collection)
        table_client = self._table_client.create_table_if_not_exists(table_name)
        entities = [
            {
                "PartitionKey": partition_key,
                "RowKey": key,
                **self._serialize(val),
            }
            for key, val in kv_pairs
        ]
        for batch in (
            entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
        ):
            table_client.submit_transaction([("upsert", entity) for entity in batch])

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> None:
        """Put multiple key-value pairs into the store asynchronously.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            collection (str): table name
            batch_size (int): batch size
        """
        table_name = self._sanitize_table_name(collection)
        self._check_async_client()
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        entities = [
            {
                "PartitionKey": partition_key,
                "RowKey": key,
                **self._serialize(val),
            }
            for key, val in kv_pairs
        ]
        for batch in (
            entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
        ):
            await table_client.submit_transaction(
                [("upsert", entity) for entity in batch]
            )

    def get(
        self,
        key: str,
        collection: str = DEFAULT_COLLECTION,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): table name
        """
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = self._sanitize_table_name(collection)
        table_client = self._table_client.create_table_if_not_exists(table_name)
        try:
            entity = table_client.get_entity(partition_key=partition_key, row_key=key)
            return self._deserialize(entity)
        except ResourceNotFoundError:
            return None

    async def aget(
        self,
        key: str,
        collection: str = DEFAULT_COLLECTION,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> Optional[dict]:
        """Get a value from the store asynchronously.

        Args:
            key (str): key
            collection (str): table name
        """
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = self._sanitize_table_name(collection)
        self._check_async_client()
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        try:
            entity = await table_client.get_entity(
                partition_key=partition_key, row_key=key
            )
            return self._deserialize(entity)
        except ResourceNotFoundError:
            return None

    def get_all(
        self,
        collection: str = DEFAULT_COLLECTION,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (str): table name
        """
        table_name = self._sanitize_table_name(collection)
        table_client = self._table_client.create_table_if_not_exists(table_name)
        entities = table_client.list_entities(
            filter=f"PartitionKey eq '{partition_key}'"
        )
        return {entity["RowKey"]: self._deserialize(entity) for entity in entities}

    async def aget_all(
        self,
        collection: str = DEFAULT_COLLECTION,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> Dict[str, dict]:
        """Get all values from the store asynchronously.

        Args:
            collection (str): table name
        """
        table_name = self._sanitize_table_name(collection)
        self._check_async_client()
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        entities = table_client.list_entities(
            filter=f"PartitionKey eq '{partition_key}'"
        )
        return {
            entity["RowKey"]: self._deserialize(entity) async for entity in entities
        }

    def delete(
        self,
        key: str,
        collection: str = DEFAULT_COLLECTION,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> bool:
        """Delete a value from the store. Always returns True.

        Args:
            key (str): key
            collection (str): table name
        """
        table_name = self._sanitize_table_name(collection)
        table_client = self._table_client.create_table_if_not_exists(table_name)
        table_client.delete_entity(partition_key=partition_key, row_key=key)
        return True

    async def adelete(
        self,
        key: str,
        collection: str = DEFAULT_COLLECTION,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> bool:
        """Delete a value from the store asynchronously. Always returns True.

        Args:
            key (str): key
            collection (str): table name
        """
        table_name = self._sanitize_table_name(collection)
        self._check_async_client()
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        await table_client.delete_entity(partition_key=partition_key, row_key=key)
        return True

    @classmethod
    def _create_clients(
        cls, endpoint: str, credential: Any, *args: Any, **kwargs: Any
    ) -> "AzureKVStore":
        """Private method to create synchronous and asynchronous table service clients.

        Args:
            endpoint (str): The service endpoint.
            credential (Any): Credentials used to authenticate requests.

        Returns:
            AzureKVStore: An instance of AzureKVStore with initialized clients.
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

    def _sanitize_table_name(self, table_name: str) -> str:
        """Sanitize table name to meet Azure Table Storage Name requirements.
        Table names must be alphanumeric, cannot begin with a number, and must be between 3 and 63 characters long.
        https://learn.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model#table-names

        Args:
            table_name (str): Table name to sanitize.

        Returns:
            str: Sanitized table name that conforms to Azure rules.
        """
        sanitized_name = ALPHANUMERIC_REGEX.sub("", table_name)
        if not sanitized_name[0].isalpha():
            sanitized_name = f"A{sanitized_name}"
        sanitized_name = sanitized_name[:63]

        if not VALID_TABLE_NAME_REGEX.match(sanitized_name):
            raise ValueError(
                "Invalid table name after sanitization. Table names must be alphanumeric, "
                "cannot begin with a number, and must be between 3 and 63 characters long."
            )
        return sanitized_name

    def _serialize(self, value: dict) -> dict:
        """Serialize all values in a dictionary to JSON strings to ensure compatibility with Azure Table Storage.
        The Azure Table Storage API does not support complex data types like dictionaries or nested objects
        directly as values in entity properties, so we need to serialize them to JSON strings.

        Args:
            val (dict): Dictionary containing the values to serialize.

        Returns:
            dict: A dictionary with all values serialized as JSON strings.
        """
        return {key: json.dumps(val, default=str) for key, val in value.items()}

    def _try_deserialize(
        self, value: str
    ) -> Optional[Union[dict, list, str, int, float, bool, None]]:
        """Attempt to deserialize a string from JSON format.
        This method tries to parse a string as JSON. If the string is valid JSON, it returns the deserialized
        Python object which could be a dictionary, list, string, integer, float, boolean, or None. If the string
        is not a valid JSON, it returns None.

        Args:
            value (str): String to check.

        Returns:
            Optional[Union[dict, list, str, int, float, bool, None]]: The Python object parsed from the JSON string,
            or None if deserialization fails.
        """
        try:
            return json.loads(value)
        except ValueError:
            return None

    def _deserialize(self, entity: dict) -> dict:
        """Deserialize values in a dictionary from JSON strings back to their original Python data types.
        This method handles the conversion of JSON-formatted strings stored in Azure Table Storage
        back into complex Python data types such as dictionaries. This is necessary because Azure Table
        Storage does not support complex data types directly as values in entity properties.

        Note: This method falls back to the original values when deserialization fails.

        Args:
            entity (dict): Dictionary containing the entity data with values as JSON strings.

        Returns:
            dict: A dictionary with all values deserialized back into their original Python data types, excluding 'PartitionKey' and 'RowKey'.
        """
        deserialized_entity = {}
        for key, val in entity.items():
            if key not in ["PartitionKey", "RowKey"]:
                deserialized_val = self._try_deserialize(val)
                deserialized_entity[key] = (
                    deserialized_val if deserialized_val is not None else val
                )
        return deserialized_entity
