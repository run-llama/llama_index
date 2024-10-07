import logging
from typing import Any, Dict, List, Optional

from azure.cosmos import CosmosClient, DatabaseProxy, ContainerProxy
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import BaseChatStore

DEFAULT_CHAT_DATABASE = "ChatMessagesDB"
DEFAULT_CHAT_CONTAINER = "ChatMessagesContainer"


logger = logging.getLogger(__name__)


# Convert a ChatMessage to a JSON object
def _message_to_dict(message: ChatMessage) -> dict:
    return message.dict()


# Convert a list of ChatMessages to a list of JSON objects
def _messages_to_dict(messages: List[ChatMessage]) -> List[dict]:
    return [_message_to_dict(message) for message in messages]


# Convert a JSON object to a ChatMessage
def _dict_to_message(d: dict) -> ChatMessage:
    return ChatMessage.model_validate(d)


class AzureCosmosNoSqlChatStore(BaseChatStore):
    """Creates an Azure Cosmos DB NoSql Chat Store."""

    _cosmos_client = CosmosClient
    _database = DatabaseProxy
    _container = ContainerProxy

    def __init__(
        self,
        cosmos_client: CosmosClient,
        chat_db_name: str = DEFAULT_CHAT_DATABASE,
        chat_container_name: str = DEFAULT_CHAT_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(
            cosmos_client=cosmos_client,
            chat_db_name=chat_db_name,
            chat_container_name=chat_container_name,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
        )

        self._cosmos_client = cosmos_client

        # Create the database if it already doesn't exist
        self._database = self._cosmos_client.create_database_if_not_exists(
            id=chat_db_name,
            offer_throughput=cosmos_database_properties.get("offer_throughput"),
            session_token=cosmos_database_properties.get("session_token"),
            initial_headers=cosmos_database_properties.get("initial_headers"),
            etag=cosmos_database_properties.get("etag"),
            match_condition=cosmos_database_properties.get("match_condition"),
        )

        # Create the collection if it already doesn't exist
        self._container = self._database.create_container_if_not_exists(
            id=chat_container_name,
            partition_key=cosmos_container_properties["partition_key"],
            indexing_policy=cosmos_container_properties.get("indexing_policy"),
            default_ttl=cosmos_container_properties.get("default_ttl"),
            offer_throughput=cosmos_container_properties.get("offer_throughput"),
            unique_key_policy=cosmos_container_properties.get("unique_key_policy"),
            conflict_resolution_policy=cosmos_container_properties.get(
                "conflict_resolution_policy"
            ),
            analytical_storage_ttl=cosmos_container_properties.get(
                "analytical_storage_ttl"
            ),
            computed_properties=cosmos_container_properties.get("computed_properties"),
            etag=cosmos_container_properties.get("etag"),
            match_condition=cosmos_container_properties.get("match_condition"),
            session_token=cosmos_container_properties.get("session_token"),
            initial_headers=cosmos_container_properties.get("initial_headers"),
        )

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        chat_db_name: str = DEFAULT_CHAT_DATABASE,
        chat_container_name: str = DEFAULT_CHAT_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ):
        """Creates an instance of Azure Cosmos DB NoSql Chat Store using a connection string."""
        cosmos_client = CosmosClient.from_connection_string(connection_string)

        return cls(
            cosmos_client,
            chat_db_name,
            chat_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )

    @classmethod
    def from_account_and_key(
        cls,
        endpoint: str,
        key: str,
        chat_db_name: str = DEFAULT_CHAT_DATABASE,
        chat_container_name: str = DEFAULT_CHAT_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ) -> "AzureCosmosNoSqlChatStore":
        """Initializes AzureCosmosNoSqlChatStore from an endpoint url and key."""
        cosmos_client = CosmosClient(endpoint, key)
        return cls(
            cosmos_client,
            chat_db_name,
            chat_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        chat_db_name: str = DEFAULT_CHAT_DATABASE,
        chat_container_name: str = DEFAULT_CHAT_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ) -> "AzureCosmosNoSqlChatStore":
        """Creates an AzureChatStore using an Azure Active Directory token."""
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        return cls._from_clients(
            endpoint,
            credential,
            chat_db_name,
            chat_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key."""
        if not self._container:
            raise ValueError("Container not initialized")
        self._container.upsert_item(
            body={
                "id": key,
                "messages": _messages_to_dict(messages),
            }
        )

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        response = self._container.read_item(key)
        if response is not None:
            message_history = response["messages"]
        else:
            message_history = []
        return [_dict_to_message(message) for message in message_history]

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        current_messages = _messages_to_dict(self.get_messages(key))
        current_messages.append(_message_to_dict(message))

        self._container.create_item(
            body={
                "id": key,
                "messages": current_messages,
            }
        )

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete messages for a key."""
        messages_to_delete = self.get_messages(key)
        self._container.delete_item(key)
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
        items = self._container.read_all_items()
        return [item["id"] for item in items]

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AzureCosmosNoSqlChatStore"

    @classmethod
    def _from_clients(
        cls,
        endpoint: str,
        credential: Any,
        chat_db_name: str = DEFAULT_CHAT_DATABASE,
        chat_container_name: str = DEFAULT_CHAT_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ) -> "AzureCosmosNoSqlChatStore":
        """Create cosmos db service clients."""
        cosmos_client = CosmosClient(url=endpoint, credential=credential)
        return cls(
            cosmos_client,
            chat_db_name,
            chat_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )
