from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Dict, List, Set, Tuple

from llama_index.legacy.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore

IMPORT_ERROR_MSG = "`boto3` package not found, please run `pip install boto3`"


def parse_schema(table: Any) -> Tuple[str, str]:
    key_hash: str | None = None
    key_range: str | None = None

    for key in table.key_schema:
        if key["KeyType"] == "HASH":
            key_hash = key["AttributeName"]
        elif key["KeyType"] == "RANGE":
            key_range = key["AttributeName"]

    if key_hash is not None and key_range is not None:
        return key_hash, key_range
    else:
        raise ValueError("Must be a DynamoDB table with a hash key and sort key.")


def convert_float_to_decimal(obj: Any) -> Any:
    if isinstance(obj, List):
        return [convert_float_to_decimal(x) for x in obj]
    elif isinstance(obj, Set):
        return {convert_float_to_decimal(x) for x in obj}
    elif isinstance(obj, Dict):
        return {k: convert_float_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        return Decimal(str(obj))
    else:
        return obj


def convert_decimal_to_int_or_float(obj: Any) -> Any:
    if isinstance(obj, List):
        return [convert_decimal_to_int_or_float(x) for x in obj]
    elif isinstance(obj, Set):
        return {convert_decimal_to_int_or_float(x) for x in obj}
    elif isinstance(obj, Dict):
        return {k: convert_decimal_to_int_or_float(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return num if (num := int(obj)) == obj else float(obj)
    else:
        return obj


class DynamoDBKVStore(BaseKVStore):
    """DynamoDB Key-Value store.
    Stores key-value pairs in a DynamoDB Table.
    The DynamoDB Table must have both a hash key and a range key,
        and their types must be string.

    You can specify a custom URL for DynamoDB by setting the `DYNAMODB_URL`
    environment variable. This is useful if you're using a local instance of
    DynamoDB for development or testing. If `DYNAMODB_URL` is not set, the
    application will use the default AWS DynamoDB service.

    Args:
        table (Any): DynamoDB Table Service Resource
    """

    def __init__(self, table: Any):
        """Init a DynamoDBKVStore."""
        try:
            from boto3.dynamodb.conditions import Key
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._table = table
        self._boto3_key = Key
        self._key_hash, self._key_range = parse_schema(table)

    @classmethod
    def from_table_name(cls, table_name: str) -> DynamoDBKVStore:
        """Load a DynamoDBKVStore from a DynamoDB table name.

        Args:
            table_name (str): DynamoDB table name
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        # Get the DynamoDB URL from environment variable
        dynamodb_url = os.getenv("DYNAMODB_URL")

        # Create a session
        session = boto3.Session()

        # If the DynamoDB URL is set, use it as the endpoint URL
        if dynamodb_url:
            ddb = session.resource("dynamodb", endpoint_url=dynamodb_url)
        else:
            # Otherwise, let boto3 use its default configuration
            ddb = session.resource("dynamodb")
        return cls(table=ddb.Table(table_name))

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name
        """
        item = {k: convert_float_to_decimal(v) for k, v in val.items()}
        item[self._key_hash] = collection
        item[self._key_range] = key
        self._table.put_item(Item=item)

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name
        """
        raise NotImplementedError

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> dict | None:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name
        """
        resp = self._table.get_item(
            Key={self._key_hash: collection, self._key_range: key}
        )
        if (item := resp.get("Item")) is None:
            return None
        else:
            return {
                k: convert_decimal_to_int_or_float(v)
                for k, v in item.items()
                if k not in {self._key_hash, self._key_range}
            }

    async def aget(self, key: str, collection: str = DEFAULT_COLLECTION) -> dict | None:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name
        """
        raise NotImplementedError

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (str): collection name
        """
        result = {}
        last_evaluated_key = None
        is_first = True
        while last_evaluated_key is not None or is_first:
            if is_first:
                is_first = False
            option = {
                "KeyConditionExpression": self._boto3_key(self._key_hash).eq(collection)
            }
            if last_evaluated_key is not None:
                option["ExclusiveStartKey"] = last_evaluated_key
            resp = self._table.query(**option)
            for item in resp.get("Items", []):
                item.pop(self._key_hash)
                key = item.pop(self._key_range)
                result[key] = {
                    k: convert_decimal_to_int_or_float(v) for k, v in item.items()
                }
            last_evaluated_key = resp.get("LastEvaluatedKey")
        return result

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (str): collection name
        """
        raise NotImplementedError

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name
        """
        resp = self._table.delete_item(
            Key={self._key_hash: collection, self._key_range: key},
            ReturnValues="ALL_OLD",
        )

        if (item := resp.get("Attributes")) is None:
            return False
        else:
            return len(item) > 0

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name
        """
        raise NotImplementedError
