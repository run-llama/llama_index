from typing import Dict, List, Optional, Tuple
import json
import logging
import textwrap
from jinja2 import Template

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

_logger = logging.getLogger(__name__)


IMPORT_ERROR_MESSAGE = """
Error: Gel Python package is not installed.
Please install it using 'pip install gel'.
"""

NO_PROJECT_MESSAGE = """
Error: it appears that the Gel project has not been initialized.
If that's the case, please run 'gel project init' to get started.
"""

MISSING_RECORD_TYPE_TEMPLATE = """
Error: Record type {{record_type}} is missing from the Gel schema.

In order to use the LlamaIndex integration, ensure you put the following in dbschema/default.gel:

    module default {
        type {{record_type}} {
            required key: str;
            required namespace: str;
            value: json;

            constraint exclusive on ((.key, .namespace));
        }
    }

Remember that you also need to run a migration:

    $ gel migration create
    $ gel migrate

"""

try:
    import gel
except ImportError as e:
    _logger.error(IMPORT_ERROR_MESSAGE)
    raise


def format_query(text: str) -> str:
    return textwrap.dedent(text.strip())


PUT_QUERY = format_query(
    """
    insert Record {
        key := <str>$key,
        namespace := <str>$namespace,
        value := <json>$value
    } unless conflict on (.key, .namespace) else (
        update Record set {
            value := <json>$value
        }
    )
    """
)

PUT_ALL_QUERY = format_query(
    """
    with
        raw_data := <json>$data,
        namespace := <str>$namespace,
    for item in json_array_unpack(raw_data) union (
        insert Record {
            key := <str>item['key'],
            namespace := namespace,
            value := <json>item['value']
        } unless conflict on (.key, .namespace) else (
            update Record set {
                value := <json>item['value']
            }
        )
    );
    """
)

GET_QUERY = format_query(
    """
    with record := (
        select Record
        filter .key = <str>$key and .namespace = <str>$namespace
    )
    select record.value;
    """
)

GET_ALL_QUERY = format_query(
    """
    select Record {
        key,
        value
    }
    filter .namespace = <str>$namespace;
    """
)

DELETE_QUERY = format_query(
    """
    delete Record filter .key = <str>$key and .namespace = <str>$namespace;
    """
)


class GelKVStore(BaseKVStore):
    """Gel Key-Value store."""

    def __init__(self, record_type: str = "Record") -> None:
        """
        Initialize GelKVStore.

        Args:
            record_type: The name of the record type in Gel schema.

        """
        self.record_type = record_type

        self._sync_client = None
        self._async_client = None

    def get_sync_client(self):
        """
        Get or initialize a synchronous Gel client.

        Ensures the client is connected and the record type exists.

        Returns:
            A connected synchronous Gel client.

        """
        if self._async_client is not None:
            raise RuntimeError(
                "GelKVStore has already been used in async mode. "
                "If you were intentionally trying to use different IO modes at the same time, "
                "please create a new instance instead."
            )
        if self._sync_client is None:
            self._sync_client = gel.create_client()

            try:
                self._sync_client.ensure_connected()
            except gel.errors.ClientConnectionError as e:
                _logger.error(NO_PROJECT_MESSAGE)
                raise

            try:
                self._sync_client.query(f"select {self.record_type};")
            except gel.errors.InvalidReferenceError as e:
                _logger.error(
                    Template(MISSING_RECORD_TYPE_TEMPLATE).render(
                        record_type=self.record_type
                    )
                )
                raise

        return self._sync_client

    async def get_async_client(self):
        """
        Get or initialize an asynchronous Gel client.

        Ensures the client is connected and the record type exists.

        Returns:
            A connected asynchronous Gel client.

        """
        if self._sync_client is not None:
            raise RuntimeError(
                "GelKVStore has already been used in sync mode. "
                "If you were intentionally trying to use different IO modes at the same time, "
                "please create a new instance instead."
            )
        if self._async_client is None:
            self._async_client = gel.create_async_client()

            try:
                await self._async_client.ensure_connected()
            except gel.errors.ClientConnectionError as e:
                _logger.error(NO_PROJECT_MESSAGE)
                raise

            try:
                await self._async_client.query(f"select {self.record_type};")
            except gel.errors.InvalidReferenceError as e:
                _logger.error(
                    Template(MISSING_RECORD_TYPE_TEMPLATE).render(
                        record_type=self.record_type
                    )
                )
                raise

        return self._async_client

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        client = self.get_sync_client()
        client.query(
            PUT_QUERY,
            key=key,
            namespace=collection,
            value=json.dumps(val),
        )

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        client = await self.get_async_client()
        await client.query(
            PUT_QUERY,
            key=key,
            namespace=collection,
            value=json.dumps(val),
        )

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Store multiple key-value pairs in batches.

        Args:
            kv_pairs: List of (key, value) tuples to store.
            collection: Namespace for the keys.
            batch_size: Number of pairs to store in each batch.

        """
        for chunk in (
            kv_pairs[pos : pos + batch_size]
            for pos in range(0, len(kv_pairs), batch_size)
        ):
            client = self.get_sync_client()
            client.query(
                PUT_ALL_QUERY,
                data=json.dumps([{"key": key, "value": value} for key, value in chunk]),
                namespace=collection,
            )

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Async version of put_all.

        Args:
            kv_pairs: List of (key, value) tuples to store.
            collection: Namespace for the keys.
            batch_size: Number of pairs to store in each batch.

        """
        for chunk in (
            kv_pairs[pos : pos + batch_size]
            for pos in range(0, len(kv_pairs), batch_size)
        ):
            client = await self.get_async_client()
            await client.query(
                PUT_ALL_QUERY,
                data=json.dumps([{"key": key, "value": value} for key, value in chunk]),
                namespace=collection,
            )

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        client = self.get_sync_client()
        result = client.query_single(
            GET_QUERY,
            key=key,
            namespace=collection,
        )
        return json.loads(result) if result is not None else None

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        client = await self.get_async_client()
        result = await client.query_single(
            GET_QUERY,
            key=key,
            namespace=collection,
        )
        return json.loads(result) if result is not None else None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        client = self.get_sync_client()
        results = client.query(
            GET_ALL_QUERY,
            namespace=collection,
        )
        return {result.key: json.loads(result.value) for result in results}

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        client = await self.get_async_client()
        results = await client.query(
            GET_ALL_QUERY,
            namespace=collection,
        )
        return {result.key: json.loads(result.value) for result in results}

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        client = self.get_sync_client()
        result = client.query(
            DELETE_QUERY,
            key=key,
            namespace=collection,
        )
        return len(result) > 0

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        client = await self.get_async_client()
        result = await client.query(
            DELETE_QUERY,
            key=key,
            namespace=collection,
        )
        return len(result) > 0
