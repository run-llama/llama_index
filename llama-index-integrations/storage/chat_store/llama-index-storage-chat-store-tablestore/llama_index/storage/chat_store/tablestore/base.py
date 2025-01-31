import json
import logging
from typing import Any, List, Optional

import tablestore
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore

logger = logging.getLogger(__name__)


# Convert a list of ChatMessages to a list of JSON objects
def _messages_to_dict(messages: List[ChatMessage]) -> List[dict]:
    return [_message_to_dict(message) for message in messages]


# Convert a ChatMessage to a JSON object
def _message_to_dict(message: ChatMessage) -> dict:
    return message.dict()


# Convert a JSON object to a ChatMessage
def _dict_to_message(d: dict) -> ChatMessage:
    return ChatMessage.model_validate(d)


class TablestoreChatStore(BaseChatStore):
    """Tablestore Chat Store.

    Args:
        tablestore_client (OTSClient, optional): External tablestore(ots) client.
                If this parameter is set, the following endpoint/instance_name/access_key_id/access_key_secret will be ignored.
        endpoint (str, optional): Tablestore instance endpoint.
        instance_name (str, optional): Tablestore instance name.
        access_key_id (str, optional): Aliyun access key id.
        access_key_secret (str, optional): Aliyun access key secret.
        table_name (str, optional): Tablestore table name.

    Returns:
        TablestoreChatStore: A Tablestore chat store object.
    """

    table_name: str
    _primary_key: str = "session_id"
    _history_column: str = "history"
    _tablestore_client: tablestore.OTSClient

    def __init__(
        self,
        tablestore_client: Optional[tablestore.OTSClient] = None,
        endpoint: Optional[str] = None,
        instance_name: Optional[str] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        table_name: str = "llama_index_chat_store_v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            table_name=table_name,
        )
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

    def create_table_if_not_exist(self) -> None:
        """Create table if not exist."""
        table_list = self._tablestore_client.list_table()
        if self.table_name in table_list:
            logger.info(
                f"Tablestore chat store table[{self.table_name}] already exists"
            )
            return
        logger.info(
            f"Tablestore chat store table[{self.table_name}] does not exist, try to create the table."
        )

        table_meta = tablestore.TableMeta(
            self.table_name, [(self._primary_key, "STRING")]
        )
        reserved_throughput = tablestore.ReservedThroughput(
            tablestore.CapacityUnit(0, 0)
        )
        self._tablestore_client.create_table(
            table_meta, tablestore.TableOptions(), reserved_throughput
        )
        logger.info(
            f"Tablestore create chat store table[{self.table_name}] successfully."
        )

    def clear_store(self):
        """Delete all messages."""
        keys = self.get_keys()
        for key in keys:
            self.delete_messages(key)

    @classmethod
    def class_name(self) -> str:
        return "TablestoreChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Assign all provided messages to the row with the given key.
        Any pre-existing messages for that key will be overwritten.

        Args:
            key (str): The key specifying a row.
            messages (List[ChatMessage]): The messages to assign to the key.

        Returns:
            None
        """
        primary_key = [(self._primary_key, key)]
        attribute_columns = [
            (
                self._history_column,
                json.dumps(_messages_to_dict(messages), ensure_ascii=False),
            ),
        ]
        row = tablestore.Row(primary_key, attribute_columns)
        self._tablestore_client.put_row(self.table_name, row)

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Retrieve all messages for the given key.

        Args:
            key (str): The key specifying a row.

        Returns:
            List[ChatMessage]: The messages associated with the key.
        """
        primary_key = [(self._primary_key, key)]
        _, row, _ = self._tablestore_client.get_row(
            self.table_name, primary_key, None, None, 1
        )
        history = {}
        if row is not None:
            for col in row.attribute_columns:
                key = col[0]
                val = col[1]
                if key == self._history_column:
                    history = json.loads(val)
                    continue
        return [_dict_to_message(message) for message in history]

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message to the end of the chat history for the given key.
        Creates a new row if the key does not exist.

        Args:
            key (str): The key specifying a row.
            message (ChatMessage): The message to add to the chat history.

        Returns:
            None
        """
        current_messages = self.get_messages(key)
        current_messages.append(message)
        self.set_messages(key, current_messages)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Deletes the entire chat history for the given key (i.e. the row).

        Args:
            key (str): The key specifying a row.

        Returns:
            Optional[List[ChatMessage]]: The messages that were deleted. None if the
                deletion failed.
        """
        messages_to_delete = self.get_messages(key)
        primary_key = [(self._primary_key, key)]
        self._tablestore_client.delete_row(self.table_name, primary_key, None)
        return messages_to_delete

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Deletes the message at the given index for the given key.

        Args:
            key (str): The key specifying a row.
            idx (int): The index of the message to delete.

        Returns:
            Optional[ChatMessage]: The message that was deleted. None if the index
                did not exist.
        """
        current_messages = self.get_messages(key)
        try:
            message_to_delete = current_messages[idx]
            del current_messages[idx]
            self.set_messages(key, current_messages)
            return message_to_delete
        except IndexError:
            logger.error(
                IndexError(f"No message exists at index, {idx}, for key {key}")
            )
            return None

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Deletes the last message in the chat history for the given key.

        Args:
            key (str): The key specifying a row.

        Returns:
            Optional[ChatMessage]: The message that was deleted. None if the chat history
                was empty.
        """
        return self.delete_message(key, -1)

    def get_keys(self) -> List[str]:
        """Retrieve all keys in the table.

        Returns:
            List[str]: The keys in the table.
        """
        keys = []
        inclusive_start_primary_key = [(self._primary_key, tablestore.INF_MIN)]
        exclusive_end_primary_key = [(self._primary_key, tablestore.INF_MAX)]
        limit = 5000
        columns_to_get = []
        (
            consumed,
            next_start_primary_key,
            row_list,
            next_token,
        ) = self._tablestore_client.get_range(
            self.table_name,
            tablestore.Direction.FORWARD,
            inclusive_start_primary_key,
            exclusive_end_primary_key,
            columns_to_get,
            limit,
            max_version=1,
        )
        if row_list:
            for row in row_list:
                keys.append(row.primary_key[0][1])
        while next_start_primary_key is not None:
            inclusive_start_primary_key = next_start_primary_key
            (
                consumed,
                next_start_primary_key,
                row_list,
                next_token,
            ) = self._tablestore_client.get_range(
                self.table_name,
                tablestore.Direction.FORWARD,
                inclusive_start_primary_key,
                exclusive_end_primary_key,
                columns_to_get,
                limit,
                max_version=1,
            )
            if row_list:
                for row in row_list:
                    keys.append(row.primary_key[0][1])

        return keys
