import json
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
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
MIN_TABLE_NAME_LENGTH = 3
MAX_TABLE_NAME_LENGTH = 63
TABLE_NAME_PLACEHOLDER_CHARACTER = "A"
DEFAULT_PARTITION_KEY = "default"
# https://learn.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model#property-types
STORAGE_MAX_ITEM_PROPERTIES = 255
STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE = 65536
STORAGE_MAX_TOTAL_PROPERTIES_SIZE_BYTES = 1048576
STORAGE_PART_KEY_DELIMITER = "_part_"
# https://learn.microsoft.com/en-us/azure/cosmos-db/concepts-limits#per-item-limits
COSMOS_MAX_TOTAL_PROPERTIES_SIZE_BYTES = 2097152
ODATA_SUPPORTED_TYPES = (bytes, bool, datetime, float, UUID, int, str)
MISSING_ASYNC_CLIENT_ERROR_MSG = "AzureKVStore was not initialized with an async client"


class ServiceMode(str, Enum):
    """
    Whether the AzureKVStore operates on an Azure Table Storage or Cosmos DB.
    """

    COSMOS = "cosmos"
    STORAGE = "storage"


class AzureKVStore(BaseKVStore):
    """Provides a key-value store interface for Azure Table Storage and Cosmos
    DB. This class supports both synchronous and asynchronous operations on
    Azure Table Storage and Cosmos DB. It supports connecting to the service
    using different credentials and manages table creation and data
    serialization to conform to the storage requirements.
    """

    def __init__(
        self,
        table_client: Any,
        atable_client: Optional[Any] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
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

        super().__init__(*args, **kwargs)

        self._table_client = cast(TableServiceClient, table_client)
        self._atable_service_client = cast(
            Optional[AsyncTableServiceClient], atable_client
        )

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        service_mode: ServiceMode = ServiceMode.STORAGE,
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
        """Creates an instance of AzureKVStore using a SAS token."""
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
        Creates an instance of AzureKVStore using Azure Active Directory
        (AAD) tokens.
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
        """Inserts or replaces a key-value pair in the specified table."""
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        self._table_client.create_table_if_not_exists(table_name).upsert_entity(
            {
                "PartitionKey": partition_key,
                "RowKey": key,
                **self._serialize(val),
            },
            UpdateMode.REPLACE,
        )

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
    ) -> None:
        """Inserts or replaces a key-value pair in the specified table."""
        try:
            from azure.data.tables import UpdateMode
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        await self._atable_service_client.create_table_if_not_exists(
            table_name
        ).upsert_entity(
            {
                "PartitionKey": partition_key,
                "RowKey": key,
                **self._serialize(val),
            },
            mode=UpdateMode.REPLACE,
        )

    def put_all(
        self,
        kv_pairs: List[Tuple[str, Optional[dict]]],
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
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
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)

        entities = []
        for key, val in kv_pairs:
            entity = {
                "PartitionKey": partition_key,
                "RowKey": key,
            }

            if val is not None:
                serialized_val = self._serialize(val)
                entity.update(serialized_val)

            entities.append(entity)

        for batch in (
            entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
        ):
            table_client.submit_transaction(
                (TransactionOperation.UPSERT, entity) for entity in batch
            )

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, Optional[dict]]],
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
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
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )

        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )

        entities = []
        for key, val in kv_pairs:
            entity = {
                "PartitionKey": partition_key,
                "RowKey": key,
            }

            if val is not None:
                serialized_val = self._serialize(val)
                entity.update(serialized_val)

            entities.append(entity)

        for batch in (
            entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
        ):
            await atable_client.submit_transaction(
                (TransactionOperation.UPSERT, entity) for entity in batch
            )

    def get(
        self,
        key: str,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
        select: Optional[str | List[str]] = None,
    ) -> Optional[dict]:
        """Retrieves a value by key from the specified table."""
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
            entity = table_client.get_entity(
                partition_key=partition_key, row_key=key, select=select
            )
            return self._deserialize(entity)
        except ResourceNotFoundError:
            return None

    async def aget(
        self,
        key: str,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
        select: Optional[str | List[str]] = None,
    ) -> Optional[dict]:
        """Retrieves a value by key from the specified table."""
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )
        try:
            entity = await atable_client.get_entity(
                partition_key=partition_key, row_key=key, select=select
            )
            return self._deserialize(entity)
        except ResourceNotFoundError:
            return None

    def get_all(
        self,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
        select: Optional[str | List[str]] = None,
    ) -> Dict[str, dict]:
        """
        Retrieves all key-value pairs from a specified partition in the table.
        """
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        table_client = self._table_client.create_table_if_not_exists(table_name)
        entities = table_client.list_entities(
            filter=f"PartitionKey eq '{partition_key}'",
            select=select,
        )
        return {entity["RowKey"]: self._deserialize(entity) for entity in entities}

    async def aget_all(
        self,
        collection: str = None,
        partition_key: str = DEFAULT_PARTITION_KEY,
        select: Optional[str | List[str]] = None,
    ) -> Dict[str, dict]:
        """
        Retrieves all key-value pairs from a specified partition in the table.
        """
        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )
        entities = atable_client.list_entities(
            filter=f"PartitionKey eq '{partition_key}'",
            select=select,
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
        """
        Deletes a specific key-value pair from the store based on the
        provided key and partition key.
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
        """Asynchronously deletes a specific key-value pair from the store."""
        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)
        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )
        await atable_client.delete_entity(partition_key=partition_key, row_key=key)
        return True

    def query(
        self,
        query_filter: str,
        collection: str = None,
        select: Optional[str | List[str]] = None,
    ) -> Generator[dict, None, None]:
        """Retrieves a value by key from the specified table."""
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
            entities = table_client.query_entities(
                query_filter=query_filter, select=select
            )

            return (self._deserialize(entity) for entity in entities)
        except ResourceNotFoundError:
            return None

    async def aquery(
        self,
        query_filter: str,
        collection: str = None,
        select: Optional[str | List[str]] = None,
    ) -> Optional[AsyncGenerator[dict, None]]:
        """Asynchronously retrieves a value by key from the specified table."""
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if self._atable_service_client is None:
            raise ValueError(MISSING_ASYNC_CLIENT_ERROR_MSG)

        table_name = (
            DEFAULT_COLLECTION
            if not collection
            else self._sanitize_table_name(collection)
        )
        atable_client = await self._atable_service_client.create_table_if_not_exists(
            table_name
        )
        try:
            entities = atable_client.query_entities(
                query_filter=query_filter, select=select
            )

            return (self._deserialize(entity) async for entity in entities)

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

    def _sanitize_table_name(self, table_name: str) -> str:
        """
        Sanitize the table name to ensure it is valid for use in Azure Table
        Storage or Cosmos DB.
        """
        san_table_name = ALPHANUMERIC_REGEX.sub("", table_name)
        if san_table_name[0].isdigit():
            san_table_name = f"{TABLE_NAME_PLACEHOLDER_CHARACTER}{san_table_name}"
        san_length = len(san_table_name)
        if san_length < MIN_TABLE_NAME_LENGTH:
            san_table_name += TABLE_NAME_PLACEHOLDER_CHARACTER * (
                MIN_TABLE_NAME_LENGTH - san_length
            )
        elif len(san_table_name) > MAX_TABLE_NAME_LENGTH:
            san_table_name = san_table_name[:MAX_TABLE_NAME_LENGTH]
        return san_table_name

    def _should_serialize(self, value: Any) -> bool:
        """Check if a value should be serialized based on its type."""
        return not isinstance(value, ODATA_SUPPORTED_TYPES) or isinstance(value, Enum)

    def _serialize_and_encode(self, value: Any) -> Tuple[str, bytes, int]:
        """
        Serialize a value to a JSON string and encode it to UTF-16 bytes for
        storage calculations.
        """
        serialized_val = json.dumps(value)
        # Azure Table Storage checks sizes against UTF-16-encoded bytes
        bytes_val = serialized_val.encode("utf-16", errors="ignore")
        val_length = len(bytes_val)
        return serialized_val, bytes_val, val_length

    def _validate_total_property_size(self, current_size: int) -> None:
        """
        Validate the total size of all properties in an entity against the
        service limits.
        """
        if (
            self.service_mode == ServiceMode.STORAGE
            and current_size > STORAGE_MAX_TOTAL_PROPERTIES_SIZE_BYTES
        ):
            raise ValueError(
                f"The total size of all properties in an Azure Table Storage Item "
                f"cannot exceed {STORAGE_MAX_TOTAL_PROPERTIES_SIZE_BYTES / 1048576}MiB.\n"
                "Consider splitting documents into smaller parts."
            )
        elif (
            self.service_mode == ServiceMode.COSMOS
            and current_size > COSMOS_MAX_TOTAL_PROPERTIES_SIZE_BYTES
        ):
            raise ValueError(
                f"The total size of all properties in an Azure Cosmos DB Item "
                f"cannot exceed {COSMOS_MAX_TOTAL_PROPERTIES_SIZE_BYTES / 1000000}MB.\n"
                "Consider splitting documents into smaller parts."
            )

    def _compute_num_parts(self, val_length: int) -> int:
        """
        Compute the number of parts to split a large property into based on the
        maximum property value size.
        """
        return val_length // STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE + (
            1 if val_length % STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE else 0
        )

    def _validate_property_count(self, num_properties: int) -> None:
        """
        Validate the number of properties in an entity against the service
        limits.
        """
        if num_properties > STORAGE_MAX_ITEM_PROPERTIES:
            raise ValueError(
                "The number of properties in an Azure Table Storage Item "
                f"cannot exceed {STORAGE_MAX_ITEM_PROPERTIES}."
            )

    def _split_large_values(
        self, num_parts: int, bytes_val: str, item: dict, key: str
    ) -> None:
        """
        Split a large property value into multiple parts and store them in the
        item dictionary.
        """
        for i in range(num_parts):
            start_index = i * STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE
            end_index = start_index + STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE
            # Convert back from UTF-16 bytes to str after slicing safely on character boundaries
            serialized_part = bytes_val[start_index:end_index].decode(
                "utf-16", errors="ignore"
            )
            item[f"{key}{STORAGE_PART_KEY_DELIMITER}{i + 1}"] = serialized_part

    def _serialize(self, value: dict) -> dict:
        """
        Serialize all values in a dictionary to JSON strings to ensure
        compatibility with Azure Table Storage.
        """
        item = {}
        num_properties = len(value)
        total_property_size = 0
        for key, val in value.items():
            # Serialize all values for the sake of size calculation
            serialized_val, bytes_val, val_length = self._serialize_and_encode(val)

            total_property_size += val_length
            self._validate_total_property_size(total_property_size)

            # Skips serialization for native ODATA types
            if not self._should_serialize(val):
                item[key] = val
                continue

            # Unlike Azure Table Storage, Cosmos DB does not have per-property limits
            if self.service_mode != ServiceMode.STORAGE:
                continue

            # No need to split the property into parts
            if val_length < STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE:
                item[key] = serialized_val
                continue

            num_parts = self._compute_num_parts(val_length)
            num_properties += num_parts

            self._validate_property_count(num_properties)

            self._split_large_values(num_parts, bytes_val, item, key)

        return item

    def _deserialize_or_fallback(self, value: str) -> Union[Any, str]:
        """
        Deserialize a JSON string back to its original Python data type, falling
        back to the original string if deserialization fails.
        """
        try:
            # Attempt to deserialize the joined parts
            return json.loads(value)
        except ValueError:
            # Fallback to the concatenated string if deserialization fails
            return value

    def _concatenate_large_values(
        self, parts_to_assemble: dict, deserialized_item: dict
    ) -> None:
        """
        Concatenate split parts of large properties back into a single value.
        """
        for base_key, parts in parts_to_assemble.items():
            concatenated_value = "".join(parts[i] for i in sorted(parts.keys()))
            deserialized_item[base_key] = self._deserialize_or_fallback(
                concatenated_value
            )

    def _deserialize(self, item: dict) -> dict:
        """
        Deserialize values in a dictionary from JSON strings back to their
        original Python data types.
        """
        deserialized_item = {}
        parts_to_assemble = defaultdict(dict)

        for key, val in item.items():
            # Only attempt to deserialize strings
            if isinstance(val, str):
                # Deserialize non-partial values
                if (
                    self.service_mode == ServiceMode.STORAGE
                    and STORAGE_PART_KEY_DELIMITER not in key
                ):
                    deserialized_item[key] = self._deserialize_or_fallback(val)
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

        self._concatenate_large_values(parts_to_assemble, deserialized_item)

        return deserialized_item
