import logging
import urllib.parse
from abc import ABC
from typing import List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import BaseChatStore
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

APP_NAME = "Llama-Index-CDBMongoVCore-ChatStore-Python"


# Convert a ChatMessage to a JSON object
def _message_to_dict(message: ChatMessage) -> dict:
    return message.dict()


# Convert a list of ChatMessages to a list of JSON objects
def _messages_to_dict(messages: List[ChatMessage]) -> List[dict]:
    return [_message_to_dict(message) for message in messages]


# Convert a JSON object to a ChatMessage
def _dict_to_message(d: dict) -> ChatMessage:
    return ChatMessage.model_validate(d)


class AzureCosmosMongoVCoreChatStore(BaseChatStore, ABC):
    """Creates an Azure Cosmos DB NoSql Chat Store."""

    _mongo_client = MongoClient
    _database = Database
    _collection = Collection

    def __init__(
        self,
        mongo_client: MongoClient,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        super().__init__(
            mongo_client=mongo_client,
            uri=uri,
            host=host,
            port=port,
            db_name=db_name,
        )

        self._mongo_client = mongo_client
        self._uri = uri
        self._host = host
        self._port = port
        self._database = self._mongo_client[db_name]
        self._collection = self._mongo_client[db_name][collection_name]

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """Creates an instance of AzureCosmosMongoVCoreChatStore using a connection string."""
        # Parse the MongoDB URI
        parsed_uri = urllib.parse.urlparse(connection_string)
        # Extract username and password, and perform url_encoding
        username = urllib.parse.quote_plus(parsed_uri.username)
        password = urllib.parse.quote_plus(parsed_uri.password)

        encoded_conn_string = f"mongodb+srv://{username}:{password}@{parsed_uri.hostname}/?{parsed_uri.query}"
        mongo_client = MongoClient(encoded_conn_string, appname=APP_NAME)

        return cls(
            mongo_client=mongo_client,
            db_name=db_name,
            collection_name=collection_name,
        )

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "AzureCosmosMongoVCoreChatStore":
        """Initializes AzureCosmosMongoVCoreChatStore from an endpoint url and key."""
        mongo_client = MongoClient(host=host, port=port, appname=APP_NAME)

        return cls(
            mongo_client=mongo_client,
            host=host,
            port=port,
            db_name=db_name,
            collection_name=collection_name,
        )

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key."""
        self._collection.updateOne(
            {"_id": key},
            {"$set": {"messages": _messages_to_dict(messages)}},
            upsert=True,
        )

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        response = self._collection.find_one({"_id": key})
        if response is not None:
            message_history = response["messages"]
        else:
            message_history = []
        return [_dict_to_message(message) for message in message_history]

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        current_messages = _messages_to_dict(self.get_messages(key))
        current_messages.append(_message_to_dict(message))

        self._collection.insert_one(
            {
                "id": key,
                "messages": current_messages,
            }
        )

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete messages for a key."""
        messages_to_delete = self.get_messages(key)
        self._collection.delete_one({"_id": key})
        return messages_to_delete

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
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
        """Delete last message for a key."""
        return self.delete_message(key, -1)

    def get_keys(self) -> List[str]:
        """Get all keys."""
        return [doc["id"] for doc in self._collection.find({}, {"id": 1})]

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AzureCosmosMongoVCoreChatStore"
