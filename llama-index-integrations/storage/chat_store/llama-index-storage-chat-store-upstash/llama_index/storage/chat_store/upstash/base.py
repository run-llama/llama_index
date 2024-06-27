from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore

import logging
from typing import List, Optional, Any
from upstash_redis import Redis
from llama_index.core.bridge.pydantic import Field
import json

logger = logging.getLogger(__name__)


# Convert a ChatMessage to a json object for Redis
def _message_to_dict(message: ChatMessage) -> dict:
    return message.dict()


class UpstashChatStore(BaseChatStore):
    """Upstash chat store."""

    redis_client: Any = Field(description="Redis client.")
    ttl: Optional[int] = Field(default=None, description="Time to live in seconds.")

    def __init__(
        self,
        redis_url: str = "",
        redis_token: str = "",
        ttl: Optional[int] = None,
    ):
        if redis_url == "" or redis_token == "":
            raise ValueError("Please provide a valid URL and token")

        try:
            self.redis_client = Redis(url=redis_url, token=redis_token)
        except Exception as error:
            logger.error(f"Upstash Redis client could not be initiated: {error}")

        # self.ttl = ttl
        super().__init__(redis_client=self.redis_client, ttl=ttl)

    def class_name(self) -> str:
        """Get class name."""
        return "UpstashChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key."""
        self.redis_client.delete(key)
        for message in messages:
            self.add_message(key, message)

        if self.ttl:
            self.redis_client.expire(key, self.ttl)

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        items = self.redis_client.lrange(key, 0, -1)
        if len(items) == 0:
            return []

        return [ChatMessage.parse_raw(item) for item in items]

    def add_message(
        self, key: str, message: ChatMessage, idx: Optional[int] = None
    ) -> None:
        """Add a message to a key."""
        if idx is None:
            message_json = json.dumps(_message_to_dict(message))
            self.redis_client.rpush(key, message_json)
        else:
            self._insert_element_at_index(key, message, idx)

        if self.ttl:
            self.redis_client.expire(key, self.ttl)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete messages for a key."""
        self.redis_client.delete(key)
        return None

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete a message from a key."""
        try:
            deleted_message = self.redis_client.lindex(key, idx)
            if deleted_message is None:
                return None

            placeholder = f"{key}:{idx}:deleted"

            self.redis_client.lset(key, idx, placeholder)
            self.redis_client.lrem(key, 1, placeholder)
            if self.ttl:
                self.redis_client.expire(key, self.ttl)

            return deleted_message

        except Exception as e:
            logger.error(f"Error deleting message at index {idx} from {key}: {e}")
            return None

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete the last message from a key."""
        return self.redis_client.rpop(key)

    def get_keys(self) -> List[str]:
        """Get all keys."""
        return [key.decode("utf-8") for key in self.redis_client.keys("*")]

    def _insert_element_at_index(
        self, key: str, message: ChatMessage, idx: int
    ) -> List[ChatMessage]:
        """Insert a message at a specific index."""
        current_list = self.get_messages(key)
        current_list.insert(idx, message)

        self.redis_client.delete(key)

        self.set_messages(key, current_list)

        return current_list
