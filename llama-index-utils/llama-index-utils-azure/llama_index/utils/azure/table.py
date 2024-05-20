import json
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Tuple, Union
from uuid import UUID

# https://learn.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model#table-names
ALPHANUMERIC_REGEX = re.compile(r"[^A-Za-z0-9]")
MIN_TABLE_NAME_LENGTH = 3
MAX_TABLE_NAME_LENGTH = 63
TABLE_NAME_PLACEHOLDER_CHARACTER = "A"
# https://learn.microsoft.com/en-us/rest/api/storageservices/understanding-the-table-service-data-model#property-types
STORAGE_MAX_ITEM_PROPERTIES = 255
STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE = 65536
STORAGE_MAX_TOTAL_PROPERTIES_SIZE_BYTES = 1048576
STORAGE_PART_KEY_DELIMITER = "_part_"
# https://learn.microsoft.com/en-us/azure/cosmos-db/concepts-limits#per-item-limits
COSMOS_MAX_TOTAL_PROPERTIES_SIZE_BYTES = 2097152
NON_SERIALIZABLE_TYPES = (bytes, bool, datetime, float, UUID, int, str)


class ServiceMode(str, Enum):
    """
    Whether the AzureKVStore operates on an Azure Table Storage or Cosmos DB.
    """

    COSMOS = "cosmos"
    STORAGE = "storage"


def sanitize_table_name(table_name: str) -> str:
    """
    Sanitize the table name to ensure it is valid for use in Azure Table Storage
    or Cosmos DB.
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


def validate_table_property_count(num_properties: int) -> None:
    """
    Validate the number of properties in an entity against Azure Table Storage
    service limits.
    """
    if num_properties > STORAGE_MAX_ITEM_PROPERTIES:
        raise ValueError(
            "The number of properties in an Azure Table Storage Item "
            f"cannot exceed {STORAGE_MAX_ITEM_PROPERTIES}."
        )


def validate_table_total_property_size(
    service_mode: ServiceMode, current_size: int
) -> None:
    """
    Validate the total size of all properties in an entity against the
    service limits.
    """
    if (
        service_mode == ServiceMode.STORAGE
        and current_size > STORAGE_MAX_TOTAL_PROPERTIES_SIZE_BYTES
    ):
        raise ValueError(
            f"The total size of all properties in an Azure Table Storage Item "
            f"cannot exceed {STORAGE_MAX_TOTAL_PROPERTIES_SIZE_BYTES / 1048576}MiB.\n"
            "Consider splitting documents into smaller parts."
        )
    elif (
        service_mode == ServiceMode.COSMOS
        and current_size > COSMOS_MAX_TOTAL_PROPERTIES_SIZE_BYTES
    ):
        raise ValueError(
            f"The total size of all properties in an Azure Cosmos DB Item "
            f"cannot exceed {COSMOS_MAX_TOTAL_PROPERTIES_SIZE_BYTES / 1000000}MB.\n"
            "Consider splitting documents into smaller parts."
        )


def compute_table_property_part_count(val_length: int) -> int:
    """
    Compute the number of parts to split a large property into based on the
    maximum property value size.
    """
    return val_length // STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE + (
        1 if val_length % STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE else 0
    )


def should_serialize(value: Any) -> bool:
    """Check if a value should be serialized based on its type."""
    return not isinstance(value, NON_SERIALIZABLE_TYPES) or isinstance(value, Enum)


def serialize_and_encode(value: Any) -> Tuple[str, bytes, int]:
    """
    Serialize a value to a JSON string and encode it to UTF-16 bytes for
    storage calculations.
    """
    serialized_val = json.dumps(value)
    # Azure Table Storage checks sizes against UTF-16-encoded bytes
    bytes_val = serialized_val.encode("utf-16", errors="ignore")
    val_length = len(bytes_val)
    return serialized_val, bytes_val, val_length


def split_large_property_value(num_parts: int, bytes_val: str, key: str) -> dict:
    """Split a large property value into multiple parts."""
    parts = {}
    for i in range(num_parts):
        start_index = i * STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE
        end_index = start_index + STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE
        # Convert back from UTF-16 bytes to str after slicing safely on character boundaries
        serialized_part = bytes_val[start_index:end_index].decode(
            "utf-16", errors="ignore"
        )
        parts[f"{key}{STORAGE_PART_KEY_DELIMITER}{i + 1}"] = serialized_part
    return parts


def serialize(service_mode: ServiceMode, value: dict) -> dict:
    """
    Serialize all values in a dictionary to JSON strings to ensure compatibility
    with Azure Table Storage. The Azure Table Storage API does not support
    complex data types like dictionaries or nested objects directly as values in
    entity properties; they need to be serialized to JSON strings.
    """
    item = {}
    num_properties = len(value)
    size_properties = 0
    for key, val in value.items():
        # Serialize all values for the sake of size calculation
        serialized_val, bytes_val, val_length = serialize_and_encode(val)

        size_properties += val_length
        validate_table_total_property_size(service_mode, size_properties)

        # Skips serialization for non-enums and non-serializable types
        if not isinstance(val, Enum) and isinstance(val, NON_SERIALIZABLE_TYPES):
            item[key] = val
            continue

        # Unlike Azure Table Storage, Cosmos DB does not have per-property limits
        if service_mode != ServiceMode.STORAGE:
            continue

        # No need to split the property into parts
        if val_length < STORAGE_MAX_ITEM_PROPERTY_VALUE_SIZE:
            item[key] = serialized_val
            continue

        num_parts = compute_table_property_part_count(val_length)
        num_properties += num_parts

        validate_table_property_count(num_properties)

        parts = split_large_property_value(num_parts, bytes_val, item, key)
        item.update(parts)

    return item


def deserialize_or_fallback(value: str) -> Union[Any, str]:
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


def concatenate_large_values(parts_to_assemble: dict) -> dict:
    """Concatenate split parts of large properties back into a single value."""
    return {
        base_key: deserialize_or_fallback(
            "".join(parts[i] for i in sorted(parts.keys()))
        )
        for base_key, parts in parts_to_assemble.items()
    }


def deserialize(service_mode: ServiceMode, item: dict) -> dict:
    """
    Deserialize values in a dictionary from JSON strings back to their original
    Python data types. This method handles the conversion of JSON-formatted
    strings stored in Azure Table Storage back into complex Python data types
    such as dictionaries. It also handles reassembling split properties, falling
    back to the original values when deserialization fails.
    """
    deserialized_item = {}
    parts_to_assemble = defaultdict(dict)

    for key, val in item.items():
        # Only attempt to deserialize strings
        if isinstance(val, str):
            # Deserialize non-partial values
            if (
                service_mode == ServiceMode.STORAGE
                and STORAGE_PART_KEY_DELIMITER not in key
            ):
                deserialized_item[key] = deserialize_or_fallback(val)
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

    deserialized_property_values = concatenate_large_values(parts_to_assemble)
    deserialized_item.update(deserialized_property_values)

    return deserialized_item
