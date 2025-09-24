import json
import logging
from time import sleep
from typing import Any, Dict, List, Optional

import six
import tablestore
from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore

logger = logging.getLogger(__name__)


class TablestoreKVStore(BaseKVStore):
    """
    Tablestore Key-Value Store.

    Args:
        tablestore_client (OTSClient, optional): External tablestore(ots) client.
                If this parameter is set, the following endpoint/instance_name/access_key_id/access_key_secret will be ignored.
        endpoint (str, optional): Tablestore instance endpoint.
        instance_name (str, optional): Tablestore instance name.
        access_key_id (str, optional): Aliyun access key id.
        access_key_secret (str, optional): Aliyun access key secret.

    Returns:
        TablestoreKVStore: A Tablestore kv store object.

    """

    def __init__(
        self,
        tablestore_client: Optional[tablestore.OTSClient] = None,
        endpoint: Optional[str] = None,
        instance_name: Optional[str] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not tablestore_client:
            self._tablestore_client = tablestore.OTSClient(
                endpoint,
                access_key_id,
                access_key_secret,
                instance_name,
                retry_policy=tablestore.WriteRetryPolicy(),
                **kwargs,  # pass additional arguments
            )
        else:
            self._tablestore_client = tablestore_client

        self._update_collection()

    @staticmethod
    def _flatten_dict_to_json_strings(original_dict) -> dict:
        result_dict = {}
        for key, value in original_dict.items():
            if isinstance(
                value, (bool, bytearray, float, int, six.binary_type, six.text_type)
            ):
                result_dict[key] = value
            else:
                result_dict[key] = json.dumps(value, ensure_ascii=False)
        return result_dict

    def _update_collection(self) -> List[str]:
        """Update collection."""
        self._collections = self._tablestore_client.list_table()
        return self._collections

    def _create_collection_if_not_exist(self, collection: str) -> None:
        """Create table if not exist."""
        if collection in self._collections:
            return

        table_list = self._tablestore_client.list_table()
        if collection in table_list:
            logger.info(f"Tablestore kv store table[{collection}] already exists")
            return
        logger.info(
            f"Tablestore kv store table[{collection}] does not exist, try to create the table.",
        )

        table_meta = tablestore.TableMeta(collection, [("pk", "STRING")])
        reserved_throughput = tablestore.ReservedThroughput(
            tablestore.CapacityUnit(0, 0)
        )
        self._tablestore_client.create_table(
            table_meta, tablestore.TableOptions(), reserved_throughput
        )
        self._update_collection()
        sleep(5)
        logger.info(f"Tablestore create kv store table[{collection}] successfully.")

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        val = self._flatten_dict_to_json_strings(val)
        self._create_collection_if_not_exist(collection)
        primary_key = [("pk", key)]
        attribute_columns = list(val.items())
        row = tablestore.Row(primary_key, attribute_columns)
        self._tablestore_client.put_row(collection, row)

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        raise NotImplementedError

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        self._create_collection_if_not_exist(collection)
        primary_key = [("pk", key)]
        try:
            _, row, _ = self._tablestore_client.get_row(
                collection, primary_key, None, None, 1
            )
            if row is None:
                return None
            return self._parse_row(row)
        except tablestore.OTSServiceError as e:
            logger.error(
                f"get row failed, http_status:{e.get_http_status()}, error_code:{e.get_error_code()}, error_message:{e.get_error_message()}, request_id:{e.get_request_id()}"
            )
            if (
                e.get_error_code() == "OTSParameterInvalid"
                and "table not exist" in e.get_error_message()
            ):
                return None

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        raise NotImplementedError

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        self._create_collection_if_not_exist(collection)
        inclusive_start_primary_key = [("pk", tablestore.INF_MIN)]
        exclusive_end_primary_key = [("pk", tablestore.INF_MAX)]
        limit = 5000
        columns_to_get = []
        (
            consumed,
            next_start_primary_key,
            row_list,
            next_token,
        ) = self._tablestore_client.get_range(
            collection,
            tablestore.Direction.FORWARD,
            inclusive_start_primary_key,
            exclusive_end_primary_key,
            columns_to_get,
            limit,
            max_version=1,
        )
        ret_dict = {}
        self._parse_rows(ret_dict, row_list)
        while next_start_primary_key is not None:
            inclusive_start_primary_key = next_start_primary_key
            (
                consumed,
                next_start_primary_key,
                row_list,
                next_token,
            ) = self._tablestore_client.get_range(
                collection,
                tablestore.Direction.FORWARD,
                inclusive_start_primary_key,
                exclusive_end_primary_key,
                columns_to_get,
                limit,
                max_version=1,
            )
            self._parse_rows(ret_dict, row_list)

        return ret_dict

    def _parse_rows(self, return_result: dict, row_list: Optional[list]) -> None:
        if row_list:
            for row in row_list:
                ret = self._parse_row(row)
                return_result[row.primary_key[0][1]] = ret

    def _delete_rows(self, row_list: Optional[list], collection: str) -> None:
        if row_list:
            for row in row_list:
                key = row.primary_key[0][1]
                self.delete(key=key, collection=collection)

    @staticmethod
    def _parse_row(row: Any) -> dict[str, Any]:
        ret = {}
        for col in row.attribute_columns:
            k = col[0]
            v = col[1]
            if isinstance(v, str):
                try:
                    ret[k] = json.loads(v)
                    if not (isinstance(ret[k], (dict, list, tuple))):
                        ret[k] = v
                except json.JSONDecodeError:
                    ret[k] = v
            else:
                ret[k] = v
        return ret

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        raise NotImplementedError

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        primary_key = [("pk", key)]
        _, return_row = self._tablestore_client.delete_row(
            collection, primary_key, None
        )
        return True

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        raise NotImplementedError

    # noinspection DuplicatedCode
    def delete_all(self, collection: str = DEFAULT_COLLECTION) -> None:
        self._create_collection_if_not_exist(collection)
        inclusive_start_primary_key = [("pk", tablestore.INF_MIN)]
        exclusive_end_primary_key = [("pk", tablestore.INF_MAX)]
        limit = 5000
        columns_to_get = []
        (
            consumed,
            next_start_primary_key,
            row_list,
            next_token,
        ) = self._tablestore_client.get_range(
            collection,
            tablestore.Direction.FORWARD,
            inclusive_start_primary_key,
            exclusive_end_primary_key,
            columns_to_get,
            limit,
            max_version=1,
        )
        ret_dict = {}
        self._delete_rows(row_list, collection)
        while next_start_primary_key is not None:
            inclusive_start_primary_key = next_start_primary_key
            (
                consumed,
                next_start_primary_key,
                row_list,
                next_token,
            ) = self._tablestore_client.get_range(
                collection,
                tablestore.Direction.FORWARD,
                inclusive_start_primary_key,
                exclusive_end_primary_key,
                columns_to_get,
                limit,
                max_version=1,
            )
            self._delete_rows(row_list, collection)
