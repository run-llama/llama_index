import logging
from typing import Any, Dict, List, Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.storage.chat_store.base import BaseChatStore
from mypy_boto3_dynamodb import ServiceResource


logger = logging.getLogger(__name__)


def _messages_to_dict(messages: List[ChatMessage]) -> List[dict]:
    return [_message_to_dict(message) for message in messages]


# Convert a ChatMessage to a JSON object
def _message_to_dict(message: ChatMessage) -> dict:
    return message.dict()


# Convert a JSON object to a ChatMessage
def _dict_to_message(d: dict) -> ChatMessage:
    return ChatMessage.model_validate(d)


class DynamoDBChatStore(BaseChatStore):
    table_name: str = Field(description="DynamoDB table")
    session_id: str = Field(description="Developer-defined chat session ID")
    primary_key: str = Field(
        default="SessionId", description="Key to use for the table."
    )
    profile_name: Optional[str] = Field(
        description="The name of aws profile to use. If not given, then the default profile is used."
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
        description="AWS region name to use. Uses region configured in AWS CLI if not passed",
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

    def __init__(
        self,
        table_name: str,
        session_id: str,
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
            session_id=session_id,
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

    @classmethod
    def class_name(self) -> str:
        return "DynamoDBChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        try:
            from botocore.exceptions import ClientError
        except ImportError as e:
            raise ImportError(
                "Unable to import botocore, please install with `pip install botocore`."
            ) from e

        try:
            self._table.put_item(
                Item={self.primary_key: key, "History": _messages_to_dict(messages)}
            )
        except ClientError as err:
            logger.error(err)

    def get_messages(self, key: str) -> List[ChatMessage]:
        try:
            from botocore.exceptions import ClientError
        except ImportError as e:
            raise ImportError(
                "Unable to import botocore, please install with `pip install botocore`."
            ) from e

        response = None
        try:
            response = self._table.get_item(Key={self.primary_key: key})
        except ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning("No record found with session id: %s", key)
            else:
                logger.error(error)

        if response and "Item" in response:
            message_history = response["Item"]["History"]
        else:
            message_history = []

        return [_dict_to_message(message) for message in message_history]

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Append the message to the record in DynamoDB."""
        try:
            from botocore.exceptions import ClientError
        except ImportError as e:
            raise ImportError(
                "Unable to import botocore, please install with `pip install botocore`."
            ) from e

        current_messages = _messages_to_dict(self.get_messages(key))
        current_messages.append(_message_to_dict(message))

        try:
            self._table.put_item(
                Item={self.primary_key: key, "History": current_messages}
            )
        except ClientError as err:
            logger.error(err)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        messages_to_delete = self.get_messages(key)
        self._table.delete_item(Key={self.primary_key: key})
        return messages_to_delete

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        current_messages = self.get_messages(key)
        message_to_delete = current_messages[idx]
        del current_messages[idx]
        self.set_messages(key, current_messages)
        return message_to_delete

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        return self.delete_message(key, -1)

    def get_keys(self) -> List[str]:
        response = self._table.scan(ProjectionExpression="SessionId")
        keys = [item["SessionId"] for item in response["Items"]]
        while "LastEvaluatedKey" in response:
            response = self._table.scan(
                ProjectionExpression="SessionId",
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            keys.extend([item["SessionId"] for item in response["Items"]])
        return keys
