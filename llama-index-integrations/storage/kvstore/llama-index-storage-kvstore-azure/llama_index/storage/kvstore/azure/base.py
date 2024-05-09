import json
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, cast
from uuid import UUID

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
DISALLOWED_KEY_CHARS_REGEX = re.compile(r"[\x00-\x1F\x7F-\x9F/#\?\t\n\r\\]")
DEFAULT_PARTITION_KEY = "default"
# https://learn.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model#property-types
MAX_ITEM_PROPERTIES = 255
MAX_ITEM_PROPERTY_KEY_LENGTH = 255
MAX_ITEM_PROPERTY_VALUE_SIZE = 65536
MAX_ITEM_SIZE = 1048576
NON_SERIALIZABLE_TYPES = (bytes, bool, datetime, float, UUID, int, str)
BUILT_IN_KEYS = {"PartitionKey", "RowKey", "Timestamp"}
PART_KEY_DELIMITER = "_part_"


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
        """
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = DefaultAzureCredential()
        return cls._create_clients(endpoint, credential, *args, **kwargs)

    def _check_async_client(self) -> None:
        if self._atable_client is None:
            raise ValueError("AzureKVStore was not initialized with an async client")

    def put(
        self,
        key: str,
        val: dict,
        collection: str = None,
        partition_key: str = None,
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): table name
            partition_key (str): partition key to use
        """
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)
        entity = {
            "PartitionKey": sanitized_partition_key,
            "RowKey": self._sanitize_key(key),
            **self._serialize(val),
        }
        table_client.upsert_entity(entity, mode=UpdateMode.REPLACE)

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = None,
        partition_key: str = None,
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): table name
            partition_key (str): partition key to use
        """
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._check_async_client()
        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        await table_client.upsert_entity(
            {
                "PartitionKey": sanitized_partition_key,
                "RowKey": self._sanitize_key(key),
                **self._serialize(val),
            },
            mode=UpdateMode.REPLACE,
        )

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        partition_key: str = None,
    ) -> None:
        """Put multiple key-value pairs into the store.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            collection (str): table name
            batch_size (int): batch size
            partition_key (str): partition key to use
        """
        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)
        entities = [
            {
                "PartitionKey": sanitized_partition_key,
                "RowKey": self._sanitize_key(key),
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
        collection: str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        partition_key: str = None,
    ) -> None:
        """Put multiple key-value pairs into the store asynchronously.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            collection (str): table name
            batch_size (int): batch size
            partition_key (str): partition key to use
        """
        self._check_async_client()
        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        entities = [
            {
                "PartitionKey": sanitized_partition_key,
                "RowKey": self._sanitize_key(key),
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
        collection: str = None,
        partition_key: str = None,
    ) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): table name
            partition_key (str): partition key to use
        """
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)
        try:
            entity = table_client.get_entity(
                partition_key=sanitized_partition_key, row_key=key
            )
            return self._deserialize(entity)
        except ResourceNotFoundError:
            return None

    async def aget(
        self,
        key: str,
        collection: str = None,
        partition_key: str = None,
    ) -> Optional[dict]:
        """Get a value from the store asynchronously.

        Args:
            key (str): key
            collection (str): table name
            partition_key (str): partition key to use
        """
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._check_async_client()
        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        try:
            entity = await table_client.get_entity(
                partition_key=sanitized_partition_key, row_key=self._sanitize_key(key)
            )
            return self._deserialize(entity)
        except ResourceNotFoundError:
            return None

    def get_all(
        self,
        collection: str = None,
        partition_key: str = None,
    ) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (str): table name
            partition_key (str): partition key to use
        """
        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)
        entities = table_client.list_entities(
            filter=f"PartitionKey eq '{sanitized_partition_key}'"
        )
        return {entity["RowKey"]: self._deserialize(entity) for entity in entities}

    async def aget_all(
        self,
        collection: str = None,
        partition_key: str = None,
    ) -> Dict[str, dict]:
        """Get all values from the store asynchronously.

        Args:
            collection (str): table name
            partition_key (str): partition key to use
        """
        self._check_async_client()
        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        entities = table_client.list_entities(
            filter=f"PartitionKey eq '{sanitized_partition_key}'"
        )
        return {
            entity["RowKey"]: self._deserialize(entity) async for entity in entities
        }

    def delete(
        self,
        key: str,
        collection: str = None,
        partition_key: str = None,
    ) -> bool:
        """Delete a value from the store. Always returns True.

        Args:
            key (str): key
            collection (str): table name
            partition_key (str): partition key to use
        """
        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)
        table_client.delete_entity(
            partition_key=sanitized_partition_key, row_key=self._sanitize_key(key)
        )
        return True

    async def adelete(
        self,
        key: str,
        collection: str = None,
        partition_key: str = None,
    ) -> bool:
        """Delete a value from the store asynchronously. Always returns True.

        Args:
            key (str): key
            collection (str): table name
            partition_key (str): partition key to use
        """
        self._check_async_client()
        sanitized_partition_key, table_name = self._sanitize_inputs(
            collection, partition_key
        )
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        await table_client.delete_entity(
            partition_key=sanitized_partition_key, row_key=self._sanitize_key(key)
        )
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

    def _sanitize_key(self, key: str) -> str:
        """Sanitizes a key by removing or replacing characters that are not allowed in Azure Table Storage
        PartitionKey and RowKey fields, and truncates it to ensure it does not exceed the specified maximum length.

        Args:
            key (str): The original key string to sanitize.

        Returns:
            str: The sanitized and truncated key string.
        """
        if len(key) > MAX_ITEM_PROPERTY_KEY_LENGTH:
            key = key[:MAX_ITEM_PROPERTY_KEY_LENGTH]
        sanitized_key = DISALLOWED_KEY_CHARS_REGEX.sub("", key)
        return sanitized_key

    def _sanitize_inputs(
        self, collection: Optional[str], partition_key: Optional[str]
    ) -> Tuple[str, str]:
        """Sanitizes the inputs for the Azure Key-Value Store.

        Args:
            collection (str): The name of the collection.
            partition_key (str): The partition key.

        Returns:
            Tuple[str, str]: A tuple containing the sanitized partition key and table name.
        """
        sanitized_partition_key = (
            self._sanitize_key(partition_key)
            if partition_key != None
            else DEFAULT_PARTITION_KEY
        )
        table_name = (
            self._sanitize_table_name(collection)
            if collection != None
            else DEFAULT_COLLECTION
        )
        return (sanitized_partition_key, table_name)

    def _serialize(self, value: dict) -> dict:
        """Serialize all values in a dictionary to JSON strings to ensure compatibility with Azure Table Storage.
        The Azure Table Storage API does not support complex data types like dictionaries or nested objects
        directly as values in entity properties, so we need to serialize them to JSON strings.

        Args:
            value (dict): Dictionary containing the values to serialize.

        Returns:
            dict: A dictionary with all values serialized as JSON strings.
        """
        serialized_values = {}
        properties = len(value) + len(BUILT_IN_KEYS)
        for key, val in value.items():
            if not isinstance(val, Enum) and isinstance(val, NON_SERIALIZABLE_TYPES):
                serialized_values[key] = val
                continue
            serialized_val = json.dumps(val, default=str)
            # Azure Table Storage checks sizes agains UTF-16-encoded values
            bytes_val = serialized_val.encode("utf-16", errors="ignore")
            val_length = len(bytes_val)
            if val_length < MAX_ITEM_PROPERTY_VALUE_SIZE:
                serialized_values[key] = serialized_val
                continue
            num_chunks = val_length // MAX_ITEM_PROPERTY_VALUE_SIZE + (
                1 if val_length % MAX_ITEM_PROPERTY_VALUE_SIZE else 0
            )
            properties += num_chunks
            if properties > MAX_ITEM_PROPERTIES:
                raise ValueError(
                    "The number of properties in an Azure Table Storage Item cannot exceed 255."
                )
            # Split the serialized value into chunks
            for i in range(num_chunks):
                start_index = i * MAX_ITEM_PROPERTY_VALUE_SIZE
                end_index = start_index + MAX_ITEM_PROPERTY_VALUE_SIZE
                # Convert back from UTF-16 bytes to str after slicing safely on character boundaries
                serialized_part = bytes_val[start_index:end_index].decode(
                    "utf-16", errors="ignore"
                )
                serialized_values[f"{key}{PART_KEY_DELIMITER}{i + 1}"] = serialized_part
        return serialized_values

    def _deserialize(self, entity: dict) -> dict:
        """Deserialize values in a dictionary from JSON strings back to their original Python data types.
        This method handles the conversion of JSON-formatted strings stored in Azure Table Storage
        back into complex Python data types such as dictionaries. It also handles reassembling split properties.

        Note: This method falls back to the original values when deserialization fails.

        Args:
            entity (dict): Dictionary containing the entity data with values as JSON strings.

        Returns:
            dict: A dictionary with all values deserialized back into their original Python data types, excluding built-in keys like 'PartitionKey', 'RowKey', and 'Timestamp'.
        """
        deserialized_entity = {}
        parts_to_assemble = defaultdict(dict)
        for key, val in entity.items():
            if key in BUILT_IN_KEYS:
                continue
            # Only attempt to deserialize strings
            if isinstance(val, str):
                if PART_KEY_DELIMITER in key:
                    base_key, part_idx = key.rsplit(PART_KEY_DELIMITER, 1)
                    try:
                        converted_idx = int(part_idx)
                        parts_to_assemble[base_key][converted_idx] = val
                    except ValueError:
                        pass
                    continue
                try:
                    # Deserialize non-split values
                    deserialized_entity[key] = json.loads(val)
                except ValueError:
                    deserialized_entity[key] = val
                continue
            deserialized_entity[key] = val
        for base_key, parts in parts_to_assemble.items():
            concatenated_value = "".join(parts[i] for i in sorted(parts.keys()))
            try:
                # Attempt to deserialize the joined parts
                deserialized_entity[base_key] = json.loads(concatenated_value)
            except ValueError:
                deserialized_entity[base_key] = concatenated_value
        return deserialized_entity
