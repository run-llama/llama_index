import json
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum, auto
from itertools import chain
from typing import Any, List, Optional, Tuple, Union
from uuid import UUID

# try:
from azure.core.credentials import AzureNamedKeyCredential, AzureSasCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.data.tables import (
    TableClient,
    TableServiceClient,
    TransactionOperation,
    UpdateMode,
)
from azure.data.tables.aio import TableServiceClient as AsyncTableServiceClient
from azure.identity import DefaultAzureCredential

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore

# except ImportError:
#     raise ImportError(
#         "`azure-data-tables` package not found, please run `pip install azure-data-tables`"
#     )


DEFAULT_CHAT_TABLE = "ChatMessages"
DEFAULT_METADATA_TABLE = "ChatMetadata"
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
MISSING_ASYNC_CLIENT_ERROR_MSG = (
    "AzureChatStore was not initialized with an async client"
)


class ServiceMode(Enum):
    """Whether the AzureKVStore operates on an Azure Table Storage or Cosmos DB.

    Args:
        Enum (Enum): The enumeration type for the service mode.
    """

    COSMOS = auto()
    STORAGE = auto()


class AzureChatStore(BaseChatStore):
    """Azure chat store leveraging Azure Table Storage or Cosmos DB."""

    _table_service_client: TableServiceClient = PrivateAttr()
    _atable_service_client: AsyncTableServiceClient = PrivateAttr()

    chat_table_name: str = Field(default=DEFAULT_CHAT_TABLE)
    metadata_table_name: str = Field(default=DEFAULT_METADATA_TABLE)
    metadata_partition_key: str = Field(default=None)
    service_mode: ServiceMode = Field(default=ServiceMode.STORAGE)

    def __init__(
        self,
        table_service_client: TableServiceClient,
        atable_service_client: Optional[AsyncTableServiceClient] = None,
        chat_table_name: str = DEFAULT_CHAT_TABLE,
        metadata_table_name: str = DEFAULT_METADATA_TABLE,
        metadata_partition_key: str = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
    ):
        sanitized_chat_table_name = self._sanitize_table_name(chat_table_name)

        super().__init__(
            chat_table_name=sanitized_chat_table_name,
            metadata_table_name=self._sanitize_table_name(metadata_table_name),
            metadata_partition_key=(
                sanitized_chat_table_name
                if metadata_partition_key is None
                else metadata_partition_key
            ),
            service_mode=service_mode,
        )

        self._table_service_client = table_service_client
        self._atable_service_client = atable_service_client

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        chat_table_name: str = DEFAULT_CHAT_TABLE,
        metadata_table_name: str = DEFAULT_METADATA_TABLE,
        metadata_partition_key: str = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
    ):
        """
        Creates an instance of AzureChatStore using a connection string.

        This class method initializes the AzureChatStore using a connection string that provides credentials
        and the necessary configuration to connect to an Azure Table Storage or Cosmos DB.

        Args:
            connection_string (str): The connection string that includes credentials and other connection details.
            chat_table_name (str): The name of the table to store chat messages.
            metadata_table_name (str): The name of the table to store metadata.
            metadata_partition_key (str): The partition key for the metadata table.
            service_mode (ServiceMode): Specifies the service mode, either Azure Table Storage or Cosmos DB. Default is STORAGE.

        Returns:
            AzureChatStore: An initialized AzureChatStore instance.

        Raises:
            ImportError: If the required Azure SDK libraries are not installed.
        """
        table_service_client = TableServiceClient.from_connection_string(
            connection_string
        )
        atable_service_client = AsyncTableServiceClient.from_connection_string(
            connection_string
        )

        return cls(
            table_service_client,
            atable_service_client,
            chat_table_name,
            metadata_table_name,
            metadata_partition_key,
            service_mode,
        )

    @classmethod
    def from_account_and_key(
        cls,
        account_name: str,
        account_key: str,
        endpoint: Optional[str] = None,
        chat_table_name: str = DEFAULT_CHAT_TABLE,
        metadata_table_name: str = DEFAULT_METADATA_TABLE,
        metadata_partition_key: str = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
    ) -> "AzureChatStore":
        """
        Initializes AzureChatStore from an account name and key.

        Provides a method to create an instance of AzureChatStore using the Azure Storage Account name and key,
        with an optional endpoint specification. Suitable for scenarios where a connection string is not available.

        Args:
            account_name (str): The Azure Storage Account name.
            account_key (str): The Azure Storage Account key.
            endpoint (Optional[str]): The specific endpoint URL for the Azure Table service. If not provided, a default is constructed.
            chat_table_name (str): The name of the table to store chat messages.
            metadata_table_name (str): The name of the table to store metadata.
            metadata_partition_key (str): The partition key for the metadata table.
            service_mode (ServiceMode): Specifies whether to use Azure Table Storage or Cosmos DB. Default is STORAGE.

        Returns:
            AzureChatStore: A configured instance of AzureChatStore.

        Raises:
            ImportError: If necessary Azure SDK components are not installed.
        """
        if endpoint is None:
            endpoint = f"https://{account_name}.table.core.windows.net"
        credential = AzureNamedKeyCredential(account_name, account_key)
        return cls._from_clients(
            endpoint,
            credential,
            chat_table_name,
            metadata_table_name,
            metadata_partition_key,
            service_mode,
        )

    @classmethod
    def from_sas_token(
        cls,
        endpoint: str,
        sas_token: str,
        chat_table_name: str = DEFAULT_CHAT_TABLE,
        metadata_table_name: str = DEFAULT_METADATA_TABLE,
        metadata_partition_key: str = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
    ) -> "AzureChatStore":
        """
        Creates an AzureChatStore instance using a SAS token.

        This method allows initializing the store with a Shared Access Signature (SAS) token, which provides
        restricted access to the storage service without exposing account keys.

        Args:
            endpoint (str): The Azure Table service endpoint URL.
            sas_token (str): The Shared Access Signature token providing limited permissions.
            chat_table_name (str): The name of the table to store chat messages.
            metadata_table_name (str): The name of the table to store metadata.
            metadata_partition_key (str): The partition key for the metadata table.
            service_mode (ServiceMode): Determines if the store operates on Azure Table Storage or Cosmos DB. Default is STORAGE.

        Returns:
            AzureChatStore: An instance of AzureChatStore configured with a SAS token.

        Raises:
            ImportError: If the required libraries are not installed.
        """
        credential = AzureSasCredential(sas_token)
        return cls._from_clients(
            endpoint,
            credential,
            chat_table_name,
            metadata_table_name,
            metadata_partition_key,
            service_mode,
        )

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        chat_table_name: str = DEFAULT_CHAT_TABLE,
        metadata_table_name: str = DEFAULT_METADATA_TABLE,
        metadata_partition_key: str = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
    ) -> "AzureChatStore":
        """
        Initializes AzureChatStore using Azure Active Directory (AAD) tokens.

        This constructor is suited for environments where AAD authentication is preferred for interacting with Azure services.
        It uses the default credentials obtained through the environment or managed identity.

        Args:
            endpoint (str): The endpoint URL for the Azure Table service.
            chat_table_name (str): The name of the table to store chat messages.
            metadata_table_name (str): The name of the table to store metadata.
            metadata_partition_key (str): The partition key for the metadata table.
            service_mode (ServiceMode): Specifies the operational mode, either Azure Table Storage or Cosmos DB. Default is STORAGE.

        Returns:
            AzureChatStore: A new AzureChatStore instance authenticated via AAD.

        Raises:
            ImportError: If necessary Azure SDK components are not installed.
        """
        credential = DefaultAzureCredential()
        return cls._from_clients(
            endpoint,
            credential,
            chat_table_name,
            metadata_table_name,
            metadata_partition_key,
            service_mode,
        )

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key."""
        # Delete existing messages and insert new messages in one transaction
        chat_client = self._table_service_client.create_table_if_not_exists(
            self.chat_table_name
        )
        entities = chat_client.query_entities(f"PartitionKey eq '{key}'")
        delete_operations = (
            (TransactionOperation.DELETE, entity) for entity in entities
        )
        create_operations = (
            (
                TransactionOperation.CREATE,
                self._serialize(
                    {
                        "PartitionKey": key,
                        "RowKey": self._to_row_key(idx),
                        **message.dict(),
                    }
                ),
            )
            for idx, message in enumerate(messages)
        )
        chat_client.submit_transaction(chain(delete_operations, create_operations))

        # Update metadata
        metadata_client = self._table_service_client.create_table_if_not_exists(
            self.metadata_table_name
        )
        messages_len = len(messages)
        metadata_client.upsert_entity(
            {
                "PartitionKey": self.metadata_partition_key,
                "RowKey": key,
                "LastMessageRowKey": self._to_row_key(messages_len - 1),
                "MessageCount": messages_len,
            },
            UpdateMode.REPLACE,
        )

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        chat_client = self._table_service_client.create_table_if_not_exists(
            self.chat_table_name
        )
        entities = chat_client.query_entities(f"PartitionKey eq '{key}'")
        return [ChatMessage.parse_obj(self._deserialize(entity)) for entity in entities]

    def add_message(self, key: str, message: ChatMessage, idx: int = None):
        """Add a message for a key."""
        # Fetch current metadata to find the next index
        metadata_client = self._table_service_client.create_table_if_not_exists(
            self.metadata_table_name
        )
        metadata = self._get_or_default_metadata(metadata_client, key)
        next_index = int(metadata["MessageCount"])

        if idx is not None and idx > next_index:
            raise ValueError(f"Index out of bounds: {idx}")
        elif idx is None:
            idx = next_index

        # Insert the new message
        chat_client = self._table_service_client.create_table_if_not_exists(
            self.chat_table_name
        )
        chat_client.create_entity(
            self._serialize(
                {
                    "PartitionKey": key,
                    "RowKey": self._to_row_key(idx),
                    **message.dict(),
                }
            )
        )

        metadata["LastMessageRowKey"] = self._to_row_key(idx)
        metadata["MessageCount"] = next_index + 1
        # Update medatada
        metadata_client.upsert_entity(metadata, UpdateMode.MERGE)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        # Delete all messages for the key
        chat_client = self._table_service_client.create_table_if_not_exists(
            self.chat_table_name
        )
        entities = chat_client.query_entities(f"PartitionKey eq '{key}'")
        chat_client.submit_transaction(
            (TransactionOperation.DELETE, entity) for entity in entities
        )

        # Reset metadata
        metadata_client = self._table_service_client.create_table_if_not_exists(
            self.metadata_table_name
        )
        metadata_client.upsert_entity(
            self._get_default_metadata(key), UpdateMode.REPLACE
        )

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
        # Fetch metadata to get the message count
        metadata_client = self._table_service_client.create_table_if_not_exists(
            self.metadata_table_name
        )
        metadata = metadata_client.get_entity(
            partition_key=self.metadata_partition_key, row_key=key
        )

        # Index out of bounds
        message_count = int(metadata["MessageCount"])
        if idx >= message_count:
            return None

        # Delete the message
        chat_client = self._table_service_client.create_table_if_not_exists(
            self.chat_table_name
        )
        chat_client.delete_entity(partition_key=key, row_key=self._to_row_key(idx))

        # Update metadata if last message was deleted
        if idx == message_count - 1:
            metadata["LastMessageRowKey"] = self._to_row_key(idx - 1)
            metadata["MessageCount"] = message_count - 1
            metadata_client.upsert_entity(metadata, mode=UpdateMode.MERGE)

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete last message for a key."""
        metadata_client = self._table_service_client.create_table_if_not_exists(
            self.metadata_table_name
        )
        # Retrieve metadata to get the last message row key
        metadata = metadata_client.get_entity(
            partition_key=self.metadata_partition_key, row_key=key
        )
        last_row_key = metadata["LastMessageRowKey"]

        chat_client = self._table_service_client.create_table_if_not_exists(
            self.chat_table_name
        )
        # Delete the last message
        chat_client.delete_entity(partition_key=key, row_key=last_row_key)

        # Update metadata
        last_row_key_num = int(last_row_key)
        metadata["LastMessageRowKey"] = self._to_row_key(
            last_row_key_num - 1 if last_row_key_num > 0 else 0
        )
        metadata["MessageCount"] = int(metadata["MessageCount"]) - 1
        metadata_client.upsert_entity(metadata, UpdateMode.MERGE)

    def get_keys(self) -> List[str]:
        """Get all keys."""
        metadata_client = self._table_service_client.create_table_if_not_exists(
            self.metadata_table_name
        )
        entities = metadata_client.query_entities(
            f"PartitionKey eq '{self.metadata_partition_key}'"
        )
        return [entity["RowKey"] for entity in entities]

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AzureChatStore"

    @classmethod
    def _from_clients(
        cls,
        endpoint: str,
        credential: Any,
        chat_table_name: str = DEFAULT_CHAT_TABLE,
        metadata_table_name: str = DEFAULT_METADATA_TABLE,
        metadata_partition_key: str = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
    ) -> "AzureChatStore":
        """Private method to create synchronous and asynchronous table service clients.

        Args:
            endpoint (str): The service endpoint.
            credential (Any): Credentials used to authenticate requests.

        Returns:
            AzureChatStore: An instance of AzureChatStore with initialized clients.
        """
        table_service_client = TableServiceClient(
            endpoint=endpoint, credential=credential
        )
        atable_service_client = AsyncTableServiceClient(
            endpoint=endpoint, credential=credential
        )

        return cls(
            table_service_client,
            atable_service_client,
            chat_table_name,
            metadata_table_name,
            metadata_partition_key,
            service_mode,
        )

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
            # Prepend an 'A' character to meet alpha character restriction
            san_table_name = f"A{san_table_name}"
        san_length = len(san_table_name)
        if san_length < 3:
            # Pad with 'A' characters to meet the minimum length requirement
            san_table_name += "A" * (3 - san_length)
        return san_table_name

    def _to_row_key(self, idx: int) -> str:
        """Generate a row key from an index.

        Args:
            idx (int): The index to convert to a row key.

        Returns:
            str: The row key generated from the index.
        """
        return f"{idx:010}"

    def _get_default_metadata(self, key: str) -> dict:
        """Generate default metadata for a key.

        Args:
            key (str): The partition key for which to generate default metadata.

        Returns:
            dict: A dictionary containing the default metadata structure.
        """
        return {
            "PartitionKey": self.metadata_partition_key,
            "RowKey": key,
            "LastMessageRowKey": self._to_row_key(0),
            "MessageCount": 0,
        }

    def _get_or_default_metadata(self, metadata_client: TableClient, key: str) -> dict:
        """Retrieve metadata if it exists, otherwise return default metadata structure.

        Args:
            metadata_client (TableClient): The client for accessing the metadata table.
            key (str): The partition key for which to retrieve or initialize metadata.

        Returns:
            dict: A dictionary containing the metadata or a default structure if not found.
        """
        try:
            return metadata_client.get_entity(
                partition_key=self.metadata_partition_key, row_key=key
            )
        except ResourceNotFoundError:
            return self._get_default_metadata(key)

    def _serialize_and_encode(self, value: Any) -> Tuple[str, bytes, int]:
        """Serialize a value to a JSON string and encode it to UTF-16 bytes for storage calculations.

        Args:
            value (Any): The value to serialize.

        Returns:
            Tuple[str, bytes, int]: A tuple containing the serialized value as a JSON string, the UTF-16-encoded bytes, and the length of the encoded value.
        """
        serialized_val = json.dumps(value, default=str)
        # Azure Table Storage checks sizes against UTF-16-encoded bytes
        bytes_val = serialized_val.encode("utf-16", errors="ignore")
        val_length = len(bytes_val)
        return serialized_val, bytes_val, val_length

    def _validate_properties_size(self, current_size: int) -> None:
        """Validate the total size of all properties in an entity against the service limits.

        Args:
            current_size (int): The current total size of all properties in an entity.

        Raises:
            ValueError: If the total size exceeds the service limits.
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
        """Compute the number of parts to split a large property into based on the maximum property value size.

        Args:
            val_length (int): The length of the property value in bytes.

        Returns:
            int: The number of parts to split the property into.
        """
        return val_length // STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE + (
            1 if val_length % STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE else 0
        )

    def _validate_property_count(self, num_properties: int) -> None:
        """Validate the number of properties in an entity against the service limits.

        Args:
            num_properties (int): The number of properties in the entity.

        Raises:
            ValueError: If the number of properties exceeds the service limits.
        """
        if num_properties > STORAGE_MAX_ITEM_PROPERTIES:
            raise ValueError(
                "The number of properties in an Azure Table Storage Item "
                f"cannot exceed {STORAGE_MAX_ITEM_PROPERTIES}."
            )

    def _split_large_values(
        self, num_parts: int, bytes_val: str, item: dict, key: str
    ) -> None:
        """Split a large property value into multiple parts and store them in the item dictionary.

        Args:
            num_parts (int): The number of parts to split the property value into.
            bytes_val (str): The UTF-16-encoded bytes of the property value.
            item (dict): The dictionary to store the split parts.
            key (str): The key for the property value.
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
        size_properties = 0
        for key, val in value.items():
            # Serialize all values for the sake of size calculation
            serialized_val, bytes_val, val_length = self._serialize_and_encode(val)

            size_properties += val_length
            self._validate_properties_size(size_properties)

            # Skips serialization for non-enums and non-serializable types
            if not isinstance(val, Enum) and isinstance(val, NON_SERIALIZABLE_TYPES):
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
        """Deserialize a JSON string back to its original Python data type, falling
        back to the original string if deserialization fails.

        Args:
            value (str): The JSON string to deserialize.

        Returns:
            Union[Any, str]: The deserialized value or the original string if deserialization fails.
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
        """Concatenate split parts of large properties back into a single value.

        Args:
            parts_to_assemble (dict): A dictionary containing the parts of large properties to reassemble.
            deserialized_item (dict): The dictionary to store the reassembled properties.
        """
        for base_key, parts in parts_to_assemble.items():
            concatenated_value = "".join(parts[i] for i in sorted(parts.keys()))
            deserialized_item[base_key] = self._deserialize_or_fallback(
                concatenated_value
            )

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
