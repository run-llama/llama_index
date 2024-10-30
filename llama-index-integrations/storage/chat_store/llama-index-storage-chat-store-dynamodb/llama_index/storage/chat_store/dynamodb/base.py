import logging
from typing import Any, Dict, List, Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.storage.chat_store.base import BaseChatStore
from mypy_boto3_dynamodb import ServiceResource


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


class DynamoDBChatStore(BaseChatStore):
    """DynamoDB Chat Store.

    Args:
        table_name (str): The name of the preexisting DynamoDB table.
        primary_key (str, optional): The primary/partition key to use for the table.
            Defaults to "SessionId".
        profile_name (str, optional): The AWS profile to use. If not specified, then
            the default AWS profile is used.
        aws_access_key_id (str, optional): The AWS Access Key ID to use.
        aws_secret_access_key (str, optional): The AWS Secret Access Key to use.
        aws_session_token (str, optional): The AWS Session Token to use.
        botocore_session (Any, optional): Use this Botocore session instead of creating a new default one.
        botocore_config (Any, optional): Custom configuration object to use instead of the default generated one.
        region_name (str, optional): The AWS region name to use. Uses the region configured in AWS CLI if not passed.
        max_retries (int, optional): The maximum number of API retries. Defaults to 10.
        timeout (float, optional): The timeout for API requests in seconds. Defaults to 60.0.
        session_kwargs (Dict[str, Any], optional): Additional kwargs for the `boto3.Session` object.
        resource_kwargs (Dict[str, Any], optional): Additional kwargs for the `boto3.Resource` object.

    Returns:
        DynamoDBChatStore: A DynamoDB chat store object.
    """

    table_name: str = Field(description="DynamoDB table")
    primary_key: str = Field(
        default="SessionId", description="Primary/partition key to use for the table."
    )
    profile_name: Optional[str] = Field(
        description="AWS profile to use. If not specified, then the default AWS profile is used."
    )
    aws_access_key_id: Optional[str] = Field(
        description="AWS Access Key ID to use.", exclude=True
    )
    aws_secret_access_key: Optional[str] = Field(
        description="AWS Secret Access Key to use.", exclude=True
    )
    aws_session_token: Optional[str] = Field(
        description="AWS Session Token to use.", exclude=True
    )
    botocore_session: Optional[Any] = Field(
        description="Use this Botocore session instead of creating a new default one.",
        exclude=True,
    )
    botocore_config: Optional[Any] = Field(
        description="Custom configuration object to use instead of the default generated one.",
        exclude=True,
    )
    region_name: Optional[str] = Field(
        description="AWS region name to use. Uses the region configured in AWS CLI if not passed",
        exclude=True,
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", gt=0
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout for API requests in seconds.",
    )
    session_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the `boto3.Session` object.",
    )
    resource_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the `boto3.Resource` object.",
    )

    _client: ServiceResource = PrivateAttr()
    _table: Any = PrivateAttr()
    _aclient: ServiceResource = PrivateAttr()
    _atable: Any = PrivateAttr()

    def __init__(
        self,
        table_name: str,
        primary_key: str = "SessionId",
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        botocore_session: Optional[Any] = None,
        botocore_config: Optional[Any] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        session_kwargs: Optional[Dict[str, Any]] = None,
        resource_kwargs: Optional[Dict[str, Any]] = None,
    ):
        session_kwargs = session_kwargs or {}
        resource_kwargs = resource_kwargs or {}

        super().__init__(
            table_name=table_name,
            primary_key=primary_key,
            profile_name=profile_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            botocore_session=botocore_session,
            botocore_config=botocore_config,
            max_retries=max_retries,
            timeout=timeout,
            session_kwargs=session_kwargs,
            resource_kwargs=resource_kwargs,
        )

        session_kwargs = {
            "profile_name": profile_name,
            "region_name": region_name,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
            "botocore_session": botocore_session,
            **session_kwargs,
        }

        try:
            import boto3
            from botocore.config import Config

            config = (
                Config(
                    retries={"max_attempts": max_retries, "mode": "standard"},
                    connect_timeout=timeout,
                    read_timeout=timeout,
                )
                if botocore_config is None
                else botocore_config
            )
            session = boto3.Session(**session_kwargs)
        except ImportError:
            raise ImportError(
                "boto3 package not found, install with 'pip install boto3"
            )

        self._client = session.resource("dynamodb", config=config, **resource_kwargs)
        self._table = self._client.Table(table_name)

    async def init_async_table(self):
        """Initialize asynchronous table."""
        if self._atable is None:
            try:
                import aioboto3

                async_session = aioboto3.Session(**self.session_kwargs)
            except ImportError:
                raise ImportError(
                    "aioboto3 package not found, install with 'pip install aioboto3'"
                )

            async with async_session.resource(
                "dynamodb", config=self.botocore_config, **self.resource_kwargs
            ) as dynamodb:
                self._atable = await dynamodb.Table(self.table_name)

    @classmethod
    def class_name(self) -> str:
        return "DynamoDBChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Assign all provided messages to the row with the given key.
        Any pre-existing messages for that key will be overwritten.

        Args:
            key (str): The key specifying a row.
            messages (List[ChatMessage]): The messages to assign to the key.

        Returns:
            None
        """
        self._table.put_item(
            Item={self.primary_key: key, "History": _messages_to_dict(messages)}
        )

    async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
        self.init_async_table()
        await self._atable.put_item(
            Item={self.primary_key: key, "History": _messages_to_dict(messages)}
        )

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Retrieve all messages for the given key.

        Args:
            key (str): The key specifying a row.

        Returns:
            List[ChatMessage]: The messages associated with the key.
        """
        response = self._table.get_item(Key={self.primary_key: key})

        if response and "Item" in response:
            message_history = response["Item"]["History"]
        else:
            message_history = []

        return [_dict_to_message(message) for message in message_history]

    async def aget_messages(self, key: str) -> List[ChatMessage]:
        self.init_async_table()
        response = await self._atable.get_item(Key={self.primary_key: key})

        if response and "Item" in response:
            message_history = response["Item"]["History"]
        else:
            message_history = []

        return [_dict_to_message(message) for message in message_history]

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message to the end of the chat history for the given key.
        Creates a new row if the key does not exist.

        Args:
            key (str): The key specifying a row.
            message (ChatMessage): The message to add to the chat history.

        Returns:
            None
        """
        current_messages = _messages_to_dict(self.get_messages(key))
        current_messages.append(_message_to_dict(message))

        self._table.put_item(Item={self.primary_key: key, "History": current_messages})

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        self.init_async_table()
        current_messages = _messages_to_dict(await self.aget_messages(key))
        current_messages.append(_message_to_dict(message))

        await self._atable.put_item(
            Item={self.primary_key: key, "History": current_messages}
        )

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Deletes the entire chat history for the given key (i.e. the row).

        Args:
            key (str): The key specifying a row.

        Returns:
            Optional[List[ChatMessage]]: The messages that were deleted. None if the
                deletion failed.
        """
        messages_to_delete = self.get_messages(key)
        self._table.delete_item(Key={self.primary_key: key})
        return messages_to_delete

    async def adelete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        self.init_async_table()
        messages_to_delete = await self.aget_messages(key)
        await self._atable.delete_item(Key={self.primary_key: key})
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

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        self.init_async_table()
        current_messages = await self.aget_messages(key)
        try:
            message_to_delete = current_messages[idx]
            del current_messages[idx]
            await self.aset_messages(key, current_messages)
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

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        return self.adelete_message(key, -1)

    def get_keys(self) -> List[str]:
        """Retrieve all keys in the table.

        Returns:
            List[str]: The keys in the table.
        """
        response = self._table.scan(ProjectionExpression=self.primary_key)
        keys = [item[self.primary_key] for item in response["Items"]]
        while "LastEvaluatedKey" in response:
            response = self._table.scan(
                ProjectionExpression=self.primary_key,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            keys.extend([item[self.primary_key] for item in response["Items"]])
        return keys

    async def aget_keys(self) -> List[str]:
        self.init_async_table()
        response = await self._atable.scan(ProjectionExpression=self.primary_key)
        keys = [item[self.primary_key] for item in response["Items"]]
        while "LastEvaluatedKey" in response:
            response = await self._atable.scan(
                ProjectionExpression=self.primary_key,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            keys.extend([item[self.primary_key] for item in response["Items"]])
        return keys
