import json
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum, auto
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
DEFAULT_PARTITION_KEY = "default"
# https://learn.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model#property-types
STORAGE_MAX_ITEM_PROPERTIES = 255
STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE = 65536
STORAGE_MAX_TOTAL_PROPERTIES_SIZE_BYTES = 1048576
STORAGE_PART_KEY_DELIMITER = "_part_"
# https://learn.microsoft.com/en-us/azure/cosmos-db/concepts-limits#per-item-limits
COSMOS_MAX_TOTAL_PROPERTIES_SIZE_BYTES = 2097152
NON_SERIALIZABLE_TYPES = (bytes, bool, datetime, float, UUID, int, str)
BUILT_IN_KEYS = {"PartitionKey", "RowKey", "Timestamp"}
MISSING_ASYNC_CLIENT_ERROR_MSG = "AzureKVStore was not initialized with an async client"


class ServiceMode(Enum):
    """Whether the AzureKVStore operates on an Azure Table Storage or Cosmos DB.

    Args:
        Enum (Enum): The enumeration type for the service mode.
    """

    COSMOS = auto()
    STORAGE = auto()


class AzureKVStore(BaseKVStore):
    """Provides a key-value store interface for Azure Table Storage and Cosmos DB.

    This class supports both synchronous and asynchronous operations on Azure Table Storage
    and Cosmos DB, allowing for operations like put, get, delete on key-value pairs.
    It supports connecting to the service using different credentials and manages table creation
    and data serialization to conform to the storage requirements.
    """

    def __init__(
        self,
        table_client: Any,
        atable_client: Optional[Any] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the AzureKVStore with Azure Table Storage clients.

        This constructor initializes the key-value store with either Azure Table Storage or Cosmos DB clients,
        allowing for both synchronous and asynchronous operations. It also sets the service mode to dictate
        specific behaviors and limitations based on the selected storage service.

        Args:
            table_client (TableServiceClient): The client for synchronous operations, initialized externally.
            atable_client (Optional[AsyncTableServiceClient]): The client for asynchronous operations, initialized externally. Optional.
            service_mode (ServiceMode): Determines if the store operates using Azure Table Storage or Cosmos DB. Default is STORAGE.
            *args: Variable length argument list to be passed to the superclass initializer.
            **kwargs: Arbitrary keyword arguments to be passed to the superclass initializer.

        Raises:
            ImportError: If the Azure data tables package is not available.
        """
        super().__init__(*args, **kwargs)
        try:
            from azure.data.tables import TableServiceClient
            from azure.data.tables.aio import (
                TableServiceClient as AsyncTableServiceClient,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._service_mode = service_mode
        self._table_client = cast(TableServiceClient, table_client)
        self._atable_client = cast(Optional[AsyncTableServiceClient], atable_client)

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """
        Creates an instance of AzureKVStore using a connection string.

        This class method initializes the AzureKVStore using a connection string that provides credentials
        and the necessary configuration to connect to an Azure Table Storage or Cosmos DB.

        Args:
            connection_string (str): The connection string that includes credentials and other connection details.
            service_mode (ServiceMode): Specifies the service mode, either Azure Table Storage or Cosmos DB. Default is STORAGE.
            *args: Variable length argument list for further initialization.
            **kwargs: Arbitrary keyword arguments for further initialization.

        Returns:
            AzureKVStore: An initialized AzureKVStore instance.

        Raises:
            ImportError: If the required Azure SDK libraries are not installed.
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
        return cls(table_client, atable_client, service_mode, *args, **kwargs)

    @classmethod
    def from_account_and_key(
        cls,
        account_name: str,
        account_key: str,
        endpoint: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """
        Initializes AzureKVStore from an account name and key.

        Provides a method to create an instance of AzureKVStore using the Azure Storage Account name and key,
        with an optional endpoint specification. Suitable for scenarios where a connection string is not available.

        Args:
            account_name (str): The Azure Storage Account name.
            account_key (str): The Azure Storage Account key.
            endpoint (Optional[str]): The specific endpoint URL for the Azure Table service. If not provided, a default is constructed.
            service_mode (ServiceMode): Specifies whether to use Azure Table Storage or Cosmos DB. Default is STORAGE.
            *args: Additional positional arguments for initialization.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            AzureKVStore: A configured instance of AzureKVStore.

        Raises:
            ImportError: If necessary Azure SDK components are not installed.
        """
        try:
            from azure.core.credentials import AzureNamedKeyCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if endpoint is None:
            endpoint = f"https://{account_name}.table.core.windows.net"
        credential = AzureNamedKeyCredential(account_name, account_key)
        return cls._from_clients(endpoint, credential, service_mode, *args, **kwargs)

    @classmethod
    def from_sas_token(
        cls,
        endpoint: str,
        sas_token: str,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """
        Creates an AzureKVStore instance using a SAS token.

        This method allows initializing the store with a Shared Access Signature (SAS) token, which provides
        restricted access to the storage service without exposing account keys.

        Args:
            endpoint (str): The Azure Table service endpoint URL.
            sas_token (str): The Shared Access Signature token providing limited permissions.
            service_mode (ServiceMode): Determines if the store operates on Azure Table Storage or Cosmos DB. Default is STORAGE.
            *args: Extra positional arguments.
            **kwargs: Extra keyword arguments.

        Returns:
            AzureKVStore: An instance of AzureKVStore configured with a SAS token.

        Raises:
            ImportError: If the required libraries are not installed.
        """
        try:
            from azure.core.credentials import AzureSasCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = AzureSasCredential(sas_token)
        return cls._from_clients(endpoint, credential, service_mode, *args, **kwargs)

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        *args: Any,
        **kwargs: Any,
    ) -> "AzureKVStore":
        """
        Initializes AzureKVStore using Azure Active Directory (AAD) tokens.

        This constructor is suited for environments where AAD authentication is preferred for interacting with Azure services.
        It uses the default credentials obtained through the environment or managed identity.

        Args:
            endpoint (str): The endpoint URL for the Azure Table service.
            service_mode (ServiceMode): Specifies the operational mode, either Azure Table Storage or Cosmos DB. Default is STORAGE.
            *args: Additional positional arguments for constructor.
            **kwargs: Additional keyword arguments for constructor.

        Returns:
            AzureKVStore: A new AzureKVStore instance authenticated via AAD.

        Raises:
            ImportError: If necessary Azure SDK components are not installed.
        """
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        credential = DefaultAzureCredential()
        return cls._from_clients(endpoint, credential, service_mode, *args, **kwargs)

    def put(
        self,
        key: str,
        val: dict,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> None:
        """Inserts or replaces a key-value pair in the specified table.

        Args:
            key (str): The key associated with the value to store.
            val (dict): The dictionary value to store.
            collection (str, optional): The name of the table to store the value in. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for storing the value. Defaults to 'default'.

        Raises:
            ImportError: If necessary Azure SDK components are not installed.
        """
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
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
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> None:
        """Asynchronously inserts or replaces a key-value pair in the specified table.

        This method performs an asynchronous upsert operation, meaning that it will insert a new key-value pair
        or replace the existing pair if the key already exists.

        Args:
            key (str): The key associated with the value to store.
            val (dict): The dictionary value to store.
            collection (str, optional): The name of the table to store the value in. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for storing the value. Defaults to 'default'.

        Raises:
            ImportError: If the Azure data tables package is not installed.
            ValueError: If the AzureKVStore was not initialized with an asynchronous client.
        """
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
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
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Inserts or replaces multiple key-value pairs in the specified table using batch operations.

        This method groups the key-value pairs into batches of a specified size and performs
        transactional upsert operations to optimize performance and ensure atomicity.

        Args:
            kv_pairs (List[Tuple[str, dict]]): A list of tuples, where each tuple contains a key and its corresponding dictionary value.
            collection (str, optional): The name of the table to store the values in. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for storing the values. Defaults to 'default'.
            batch_size (int, optional): The number of operations to include in each transaction batch. Defaults to a sensible system-defined size.

        Raises:
            ImportError: If the Azure data tables package is not installed.
        """
        try:
            from azure.data.tables import TransactionOperation
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
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
            table_client.submit_transaction(
                [(TransactionOperation.UPSERT, entity) for entity in batch]
            )

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Asynchronously inserts or replaces multiple key-value pairs in the specified table using batch operations.

        Similar to the synchronous put_all, this method groups the key-value pairs into batches
        and performs asynchronous transactional upserts.

        Args:
            kv_pairs (List[Tuple[str, dict]]): A list of tuples, where each tuple contains a key and its corresponding dictionary value.
            collection (str, optional): The name of the table to store the values in. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for storing the values. Defaults to 'default'.
            batch_size (int, optional): The number of operations to include in each transaction batch. Defaults to a sensible system-defined size.

        Raises:
            ImportError: If the Azure data tables package is not installed.
            ValueError: If the AzureKVStore was not initialized with an asynchronous client.
        """
        try:
            from azure.data.tables import TransactionOperation
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
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
                [(TransactionOperation.UPSERT, entity) for entity in batch]
            )

    def get(
        self,
        key: str,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> Optional[dict]:
        """Retrieves a value by key from the specified table.

        This method fetches the value associated with the given key from the specified table.
        If the key is not found, it returns None.

        Args:
            key (str): The key to retrieve.
            collection (str, optional): The name of the table to retrieve the value from. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for retrieving the value. Defaults to 'default'.

        Returns:
            Optional[dict]: The dictionary value associated with the key if found, otherwise None.

        Raises:
            ImportError: If the Azure data tables package is not installed.
        """
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)
        try:
            entity = table_client.get_entity(partition_key=partition_key, row_key=key)
            return self._deserialize(entity)
        except ResourceNotFoundError:
            return None

    async def aget(
        self,
        key: str,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> Optional[dict]:
        """Asynchronously retrieves a value by key from the specified table.

        Similar to the synchronous get method, this performs an asynchronous fetch operation.

        Args:
            key (str): The key to retrieve.
            collection (str, optional): The name of the table to retrieve the value from. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for retrieving the value. Defaults to 'default'.

        Returns:
            Optional[dict]: The dictionary value associated with the key if found, otherwise None.

        Raises:
            ImportError: If the Azure data tables package is not installed.
            ValueError: If the AzureKVStore was not initialized with an asynchronous client.
        """
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
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
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> Dict[str, dict]:
        """Retrieves all key-value pairs from a specified partition in the table.

        This method fetches all entries that share the same partition key, providing a dictionary
        of key-value pairs where keys are the row keys from the storage and values are the associated
        data dictionaries.

        Args:
            collection (str, optional): The name of the table to retrieve the values from. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for retrieving the values. Defaults to 'default'.

        Returns:
            Dict[str, dict]: A dictionary containing all key-value pairs from the specified partition.

        Raises:
            ImportError: If the Azure data tables package is not installed.
        """
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)
        entities = table_client.list_entities(
            filter=f"PartitionKey eq '{partition_key}'"
        )
        return {entity["RowKey"]: self._deserialize(entity) for entity in entities}

    async def aget_all(
        self,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> Dict[str, dict]:
        """Asynchronously retrieves all key-value pairs from a specified partition in the table.

        Similar to the synchronous get_all method, this method performs an asynchronous fetch of all
        entries in the specified partition and table. It returns a dictionary of the retrieved key-value pairs.

        Args:
            collection (str, optional): The name of the table to retrieve the values from. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for retrieving the values. Defaults to 'default'.

        Returns:
            Dict[str, dict]: A dictionary containing all key-value pairs from the specified partition.

        Raises:
            ImportError: If the Azure data tables package is not installed.
            ValueError: If the AzureKVStore was not initialized with an asynchronous client.
        """
        if self._atable_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
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
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> bool:
        """Deletes a specific key-value pair from the store based on the provided key and partition key.

        This method removes a key-value pair from the specified table and partition. It always returns True,
        indicating that the operation was executed (but not necessarily that the key existed).

        Args:
            key (str): The key of the item to delete.
            collection (str, optional): The name of the table from which to delete the item. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for identifying the item. Defaults to 'default'.

        Returns:
            bool: Always returns True, signifying that the delete operation was attempted.

        Raises:
            ImportError: If the Azure data tables package is not installed.
        """
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)
        table_client.delete_entity(partition_key=partition_key, row_key=key)
        return True

    async def adelete(
        self,
        key: str,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> bool:
        """Asynchronously deletes a specific key-value pair from the store.

        Similar to the synchronous delete method, this method performs an asynchronous operation to
        remove a key-value pair. It always returns True, indicating that the operation was attempted.

        Args:
            key (str): The key of the item to delete.
            collection (str, optional): The name of the table from which to delete the item. If not specified, uses a default table.
            partition_key (str, optional): The partition key used for identifying the item. Defaults to 'default'.

        Returns:
            bool: Always returns True, signifying that the delete operation was attempted.

        Raises:
            ImportError: If the Azure data tables package is not installed.
            ValueError: If the AzureKVStore was not initialized with an asynchronous client.
        """
        if self._atable_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        table_client = self._atable_client.create_table_if_not_exists(table_name)
        await table_client.delete_entity(partition_key=partition_key, row_key=key)
        return True

    @classmethod
    def _from_clients(
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
        """Sanitize the table name to ensure it is valid for use in Azure Table Storage or Cosmos.
        Table names may contain only alphanumeric characters and cannot begin with a numeric character.
        They are case-insensitive and must be from 3 to 63 characters long.

        Args:
            table_name (str): The table name to sanitize.

        Returns:
            str: The sanitized table name.
        """
        san_table_name = ALPHANUMERIC_REGEX.sub("", table_name)
        if san_table_name[0].isdigit():
            san_table_name = f"A{san_table_name}"
        san_length = len(san_table_name)
        if san_length < 3:
            san_table_name += "A" * (3 - san_length)
        return san_table_name

    def _serialize(self, value: dict) -> dict:
        """Serialize all values in a dictionary to JSON strings to ensure compatibility with Azure Table Storage.
        The Azure Table Storage API does not support complex data types like dictionaries or nested objects
        directly as values in entity properties, so we need to serialize them to JSON strings.

        Args:
            value (dict): Dictionary containing the values to serialize.

        Returns:
            dict: A dictionary with all values serialized as JSON strings.
        """
        item = {}
        num_properties = len(value) + len(BUILT_IN_KEYS)
        total_properties_size = 0
        for key, val in value.items():
            # Serialize all values for the sake of size calculation
            serialized_val = json.dumps(val, default=str)
            # Azure Table Storage checks sizes against UTF-16-encoded bytes
            bytes_val = serialized_val.encode("utf-16", errors="ignore")
            val_length = len(bytes_val)
            total_properties_size += val_length
            if self._service_mode == ServiceMode.STORAGE:
                if total_properties_size > STORAGE_MAX_TOTAL_PROPERTIES_SIZE_BYTES:
                    raise ValueError(
                        "The total size of all properties in an Azure Table Storage Item "
                        f"cannot exceed {STORAGE_MAX_TOTAL_PROPERTIES_SIZE_BYTES / 1048576}MiB.\n"
                        "Consider splitting documents into smaller parts."
                    )
            elif self._service_mode == ServiceMode.COSMOS:
                if total_properties_size > COSMOS_MAX_TOTAL_PROPERTIES_SIZE_BYTES:
                    raise ValueError(
                        "The total size of all properties in an Azure Cosmos DB Item "
                        f"cannot exceed {COSMOS_MAX_TOTAL_PROPERTIES_SIZE_BYTES / 1000000}MB.\n"
                        "Consider splitting documents into smaller parts."
                    )

            # Skips serialization for non-enums and non-serializable types
            if not isinstance(val, Enum) and isinstance(val, NON_SERIALIZABLE_TYPES):
                item[key] = val
                continue

            # Unlike Azure Table Storage, Cosmos DB does not have per-property limits
            if self._service_mode != ServiceMode.STORAGE:
                continue

            if val_length < STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE:
                # No need to split the value
                item[key] = serialized_val
                continue

            num_parts = val_length // STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE + (
                1 if val_length % STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE else 0
            )
            num_properties += num_parts
            if num_properties > STORAGE_MAX_ITEM_PROPERTIES:
                raise ValueError(
                    "The number of properties in an Azure Table Storage Item "
                    f"cannot exceed {STORAGE_MAX_ITEM_PROPERTIES}."
                )

            # Split the serialized value into chunks
            for i in range(num_parts):
                start_index = i * STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE
                end_index = start_index + STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE
                # Convert back from UTF-16 bytes to str after slicing safely on character boundaries
                serialized_part = bytes_val[start_index:end_index].decode(
                    "utf-16", errors="ignore"
                )
                item[f"{key}{STORAGE_PART_KEY_DELIMITER}{i + 1}"] = serialized_part

        return item

    def _deserialize(self, item: dict) -> dict:
        """Deserialize values in a dictionary from JSON strings back to their original Python data types.
        This method handles the conversion of JSON-formatted strings stored in Azure Table Storage
        back into complex Python data types such as dictionaries. It also handles reassembling split properties.

        Note: This method falls back to the original values when deserialization fails.

        Args:
            item (dict): Dictionary containing the entity data with values as JSON strings.

        Returns:
            dict: A dictionary with all values deserialized back into their original Python data types, excluding built-in keys like 'PartitionKey', 'RowKey', and 'Timestamp'.
        """
        deserialized_item = {}
        parts_to_assemble = defaultdict(dict)

        for key, val in item.items():
            # Skip built-in keys
            if key in BUILT_IN_KEYS:
                continue

            # Only attempt to deserialize strings
            if isinstance(val, str):
                # Deserialize non-partial values
                if (
                    self._service_mode == ServiceMode.STORAGE
                    and STORAGE_PART_KEY_DELIMITER not in key
                ):
                    try:
                        deserialized_item[key] = json.loads(val)
                    except ValueError:
                        deserialized_item[key] = val
                    continue

                # Deserialize partial values
                base_key, part_idx = key.rsplit(STORAGE_PART_KEY_DELIMITER, 1)
                try:
                    converted_idx = int(part_idx)
                    parts_to_assemble[base_key][converted_idx] = val
                except ValueError:
                    pass
                continue

            # Assign non-serialized values
            deserialized_item[key] = val

        # Reassemble any parts
        for base_key, parts in parts_to_assemble.items():
            concatenated_value = "".join(parts[i] for i in sorted(parts.keys()))
            try:
                # Attempt to deserialize the joined parts
                deserialized_item[base_key] = json.loads(concatenated_value)
            except ValueError:
                deserialized_item[base_key] = concatenated_value
        return deserialized_item
