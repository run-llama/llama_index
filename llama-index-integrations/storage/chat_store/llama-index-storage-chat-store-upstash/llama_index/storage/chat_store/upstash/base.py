from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore

import logging
from typing import List, Optional
from upstash_redis import Redis as SyncRedis
from upstash_redis.asyncio import Redis as AsyncRedis

from llama_index.core.bridge.pydantic import Field, PrivateAttr
import json

logger = logging.getLogger(__name__)


# Convert a ChatMessage to a json object for Redis
def _message_to_dict(message: ChatMessage) -> dict:
    """
    Convert a ChatMessage to a JSON-serializable dictionary.

    Args:
        message (ChatMessage): The ChatMessage object to convert.

    Returns:
        dict: A dictionary representation of the ChatMessage.

    """
    return message.dict()


class UpstashChatStore(BaseChatStore):
    """
    Upstash chat store for storing and retrieving chat messages using Redis.

    This class implements the BaseChatStore interface and provides methods
    for managing chat messages in an Upstash Redis database.
    """

    _sync_redis_client: SyncRedis = PrivateAttr()
    _async_redis_client: AsyncRedis = PrivateAttr()

    ttl: Optional[int] = Field(default=None, description="Time to live in seconds.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        redis_url: str = "",
        redis_token: str = "",
        ttl: Optional[int] = None,
    ):
        """
        Initialize the UpstashChatStore.

        Args:
            redis_url (str): The URL of the Upstash Redis instance.
            redis_token (str): The authentication token for the Upstash Redis instance.
            ttl (Optional[int]): Time to live in seconds for stored messages.

        Raises:
            ValueError: If redis_url or redis_token is empty.

        """
        if redis_url == "" or redis_token == "":
            raise ValueError("Please provide a valid URL and token")
        super().__init__(ttl=ttl)
        try:
            self._sync_redis_client = SyncRedis(url=redis_url, token=redis_token)
            self._async_redis_client = AsyncRedis(url=redis_url, token=redis_token)
        except Exception as error:
            logger.error(f"Upstash Redis client could not be initiated: {error}")

        # self.ttl = ttl

    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name.

        Returns:
            str: The name of the class.

        """
        return "UpstashChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """
        Set messages for a key.

        Args:
            key (str): The key to store the messages under.
            messages (List[ChatMessage]): The list of messages to store.

        """
        self._sync_redis_client.delete(key)
        for message in messages:
            self.add_message(key, message)

        if self.ttl:
            self._sync_redis_client.expire(key, self.ttl)

    async def async_set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """
        Set messages for a key.

        Args:
            key (str): The key to store the messages under.
            messages (List[ChatMessage]): The list of messages to store.

        """
        await self._async_redis_client.delete(key)
        for message in messages:
            await self.async_add_message(key, message)

        if self.ttl:
            await self._async_redis_client.expire(key, self.ttl)

    def get_messages(self, key: str) -> List[ChatMessage]:
        """
        Get messages for a key.

        Args:
            key (str): The key to retrieve messages from.

        Returns:
            List[ChatMessage]: The list of retrieved messages.

        """
        items = self._sync_redis_client.lrange(key, 0, -1)
        if len(items) == 0:
            return []

        return [ChatMessage.parse_raw(item) for item in items]

    async def async_get_messages(self, key: str) -> List[ChatMessage]:
        """
        Get messages for a key.

        Args:
            key (str): The key to retrieve messages from.

        Returns:
            List[ChatMessage]: The list of retrieved messages.

        """
        items = await self._async_redis_client.lrange(key, 0, -1)
        if len(items) == 0:
            return []

        return [ChatMessage.parse_raw(item) for item in items]

    def add_message(
        self, key: str, message: ChatMessage, idx: Optional[int] = None
    ) -> None:
        """
        Add a message to a key.

        Args:
            key (str): The key to add the message to.
            message (ChatMessage): The message to add.
            idx (Optional[int]): The index at which to insert the message.

        """
        if idx is None:
            message_json = json.dumps(_message_to_dict(message))
            self._sync_redis_client.rpush(key, message_json)
        else:
            self._insert_element_at_index(key, message, idx)

        if self.ttl:
            self._sync_redis_client.expire(key, self.ttl)

    async def async_add_message(
        self, key: str, message: ChatMessage, idx: Optional[int] = None
    ) -> None:
        """
        Add a message to a key.

        Args:
            key (str): The key to add the message to.
            message (ChatMessage): The message to add.
            idx (Optional[int]): The index at which to insert the message.

        """
        if idx is None:
            message_json = json.dumps(_message_to_dict(message))
            await self._async_redis_client.rpush(key, message_json)
        else:
            await self._async_insert_element_at_index(key, message, idx)

        if self.ttl:
            await self._async_redis_client.expire(key, self.ttl)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """
        Delete messages for a key.

        Args:
            key (str): The key to delete messages from.

        Returns:
            Optional[List[ChatMessage]]: Always returns None in this implementation.

        """
        self._sync_redis_client.delete(key)
        return None

    async def async_delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """
        Delete messages for a key.

        Args:
            key (str): The key to delete messages from.

        Returns:
            Optional[List[ChatMessage]]: Always returns None in this implementation.

        """
        await self._async_redis_client.delete(key)
        return None

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """
        Delete a message from a key.

        Args:
            key (str): The key to delete the message from.
            idx (int): The index of the message to delete.

        Returns:
            Optional[ChatMessage]: The deleted message, or None if not found or an error occurred.

        """
        try:
            deleted_message = self._sync_redis_client.lindex(key, idx)
            if deleted_message is None:
                return None

            placeholder = f"{key}:{idx}:deleted"

            self._sync_redis_client.lset(key, idx, placeholder)
            self._sync_redis_client.lrem(key, 1, placeholder)
            if self.ttl:
                self._sync_redis_client.expire(key, self.ttl)

            return deleted_message

        except Exception as e:
            logger.error(f"Error deleting message at index {idx} from {key}: {e}")
            return None

    async def async_delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """
        Delete a message from a key.

        Args:
            key (str): The key to delete the message from.
            idx (int): The index of the message to delete.

        Returns:
            Optional[ChatMessage]: The deleted message, or None if not found or an error occurred.

        """
        try:
            deleted_message = await self._async_redis_client.lindex(key, idx)
            if deleted_message is None:
                return None

            placeholder = f"{key}:{idx}:deleted"

            await self._async_redis_client.lset(key, idx, placeholder)
            await self._async_redis_client.lrem(key, 1, placeholder)
            if self.ttl:
                await self._async_redis_client.expire(key, self.ttl)

            return deleted_message

        except Exception as e:
            logger.error(f"Error deleting message at index {idx} from {key}: {e}")
            return None

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """
        Delete the last message from a key.

        Args:
            key (str): The key to delete the last message from.

        Returns:
            Optional[ChatMessage]: The deleted message, or None if the list is empty.

        """
        deleted_message = self._sync_redis_client.rpop(key)
        return ChatMessage.parse_raw(deleted_message) if deleted_message else None

    async def async_delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """
        Delete the last message from a key.

        Args:
            key (str): The key to delete the last message from.

        Returns:
            Optional[ChatMessage]: The deleted message, or None if the list is empty.

        """
        deleted_message = await self._async_redis_client.rpop(key)
        if deleted_message:
            return ChatMessage.parse_raw(deleted_message)
        return None

    def get_keys(self) -> List[str]:
        """
        Get all keys.

        Returns:
            List[str]: A list of all keys in the Redis store.

        """
        keys = self._sync_redis_client.keys("*")
        return keys if isinstance(keys, list) else [keys]

    async def async_get_keys(self) -> List[str]:
        """
        Get all keys.

        Returns:
            List[str]: A list of all keys in the Redis store.

        """
        keys = await self._async_redis_client.keys("*")
        return keys if isinstance(keys, list) else [keys]

    def _insert_element_at_index(
        self, key: str, message: ChatMessage, idx: int
    ) -> List[ChatMessage]:
        """
        Insert a message at a specific index.

        Args:
            key (str): The key of the list to insert into.
            message (ChatMessage): The message to insert.
            idx (int): The index at which to insert the message.

        Returns:
            List[ChatMessage]: The updated list of messages.

        """
        current_list = self.get_messages(key)
        current_list.insert(idx, message)

        self._sync_redis_client.delete(key)

        self.set_messages(key, current_list)

        return current_list

    async def _async_insert_element_at_index(
        self, key: str, message: ChatMessage, idx: int
    ) -> List[ChatMessage]:
        """
        Insert a message at a specific index.

        Args:
            key (str): The key of the list to insert into.
            message (ChatMessage): The message to insert.
            idx (int): The index at which to insert the message.

        Returns:
            List[ChatMessage]: The updated list of messages.

        """
        current_list = await self.async_get_messages(key)
        current_list.insert(idx, message)

        await self.async_delete_messages(key)

        await self.async_set_messages(key, current_list)

        return current_list
