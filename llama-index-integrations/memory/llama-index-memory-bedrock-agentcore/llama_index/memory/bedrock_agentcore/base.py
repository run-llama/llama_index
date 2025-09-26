from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_serializer
import boto3
from botocore.config import Config

from llama_index.core.memory.memory import InsertMethod
from llama_index.core.memory.types import BaseMemory
from llama_index.memory.bedrock_agentcore.utils import (
    convert_message_to_event_payload,
    convert_messages_to_string,
    convert_memory_to_system_message,
    convert_events_to_messages,
    convert_memory_to_user_message,
)
from llama_index.core.memory import BaseMemory
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.llms.types import ChatMessage, MessageRole


class BaseAgentCoreMemory(BaseMemory):
    """Base class for Bedrock Agent Core Memory."""

    _config: Any = PrivateAttr()
    _client: Any = PrivateAttr()
    _boto_client_kwargs: Any = PrivateAttr()

    def __init__(self, client: Any, **kwargs) -> None:
        super().__init__(**kwargs)
        if client is not None:
            self._client = client

    def create_event(
        self,
        memory_id: str,
        actor_id: str,
        messages: List[ChatMessage],
        session_id: str,
        **kwargs,
    ) -> None:
        if self._client is None:
            raise ValueError("Client is not initialized")
        if len(messages) == 0:
            raise ValueError("The messages field cannot be empty")

        for message in messages:
            eventTimestamp = datetime.now(timezone.utc)
            payload = convert_message_to_event_payload(
                message=message, eventTimestamp=eventTimestamp
            )
            if payload:
                response = self._client.create_event(
                    memoryId=memory_id,
                    actorId=actor_id,
                    sessionId=session_id,
                    payload=payload,
                    eventTimestamp=eventTimestamp,
                )
                event_id = response["event"]["eventId"]
                if not event_id:
                    raise RuntimeError("Bedrock AgentCore did not return an event ID")

    def list_events(
        self, memory_id: str, session_id: str, actor_id: str
    ) -> List[ChatMessage]:
        initial_max_results = 20
        # If user is not the first message, then we need to find the closest User message to construct the oldest conversation in the batch
        iterative_max_results = 3

        next_token = None
        found_user = False

        all_messages = []
        response = self._client.list_events(
            memoryId=memory_id,
            sessionId=session_id,
            actorId=actor_id,
            includePayloads=True,
            maxResults=initial_max_results,
            **({"nextToken": next_token} if next_token else {}),
        )
        next_token = response.get("nextToken")
        messages_from_events = convert_events_to_messages(response["events"])
        messages = sorted(messages_from_events, key=lambda x: x[0], reverse=False)
        all_messages.extend(messages)

        # Check if first message is a USER msg. If it's not, some LLMs will throw an exception.
        if messages[0][1].role == MessageRole.USER:
            found_user = True

        # Call ListEvents until we find the most recent User message that constructs the full conversation for the oldest message in the initial batch
        while not found_user:
            response = self._client.list_events(
                memoryId=memory_id,
                sessionId=session_id,
                actorId=actor_id,
                includePayloads=True,
                maxResults=iterative_max_results,
                **({"nextToken": next_token} if next_token else {}),
            )
            messages = convert_events_to_messages(response["events"])

            # A 'message' is a tuple of (eventTimestamp, ChatMessage)
            # We had to create this structure because boto3 only guarantees 1 second resolution, so the workaround is to
            # Set a timestamp variable in the payload that we will use to sort the messages to guarantee order in the STM.
            for message in messages:
                if message[1].role == MessageRole.USER:
                    found_user = True
                    all_messages.extend(messages)
                    break

            # Check if there are more messages to fetch
            next_token = response.get("nextToken")
            if not next_token:
                break

            if not found_user:
                all_messages.extend(messages)

        # Sort all collected messages
        sorted_messages = [
            msg for _, msg in sorted(all_messages, key=lambda x: x[0], reverse=False)
        ]

        # Remove first message if it's not a User
        first_msg_user = False
        while not first_msg_user:
            if sorted_messages[0].role == MessageRole.USER:
                first_msg_user = True
            else:
                sorted_messages = sorted_messages[1:]

        return sorted_messages

    def retrieve_memories(
        self,
        memory_id: str,
        search_criteria: Dict[str, Any],
        max_results: int = 20,
        namespace: Optional[str] = "/",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        response = self._client.retrieve_memory_records(
            memoryId=memory_id,
            namespace=namespace,
            searchCriteria=search_criteria,
            maxResults=max_results,
        )

        memmory_record_summaries = response["memoryRecordSummaries"]

        memory_content = []
        for summary in memmory_record_summaries:
            memory_content.append(summary["content"])

        return memory_content


class AgentCoreMemoryContext(BaseModel):
    actor_id: str
    memory_id: str
    session_id: str
    namespace: str = "/"
    memory_strategy_id: Optional[str] = None

    def get_context(self) -> Dict[str, Optional[str]]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


class AgentCoreMemory(BaseAgentCoreMemory):
    search_msg_limit: int = Field(
        default=5,
        description="Limit of chat history messages to use for context in search API",
    )
    insert_method: InsertMethod = Field(
        default=InsertMethod.SYSTEM,
        description="Whether to inject memory blocks into a system message or into the latest user message.",
    )

    _context: AgentCoreMemoryContext = PrivateAttr()

    def __init__(
        self,
        context: AgentCoreMemoryContext,
        # TODO: add support for InsertMethod.USER. for now default to InsertMethod.SYSTEM
        # insert_method: InsertMethod = InsertMethod.SYSTEM,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        api_version: Optional[str] = None,
        use_ssl: bool = True,
        verify: Optional[Union[bool, str]] = None,
        endpoint_url: Optional[str] = None,
        botocore_session: Optional[Any] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 10,
        botocore_config: Optional[Any] = None,
        **kwargs,
    ) -> None:
        session_kwargs = {
            "profile_name": profile_name,
            "region_name": region_name,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
            "botocore_session": botocore_session,
        }
        self._config = (
            Config(
                retries={"max_attempts": max_retries, "mode": "standard"},
                connect_timeout=timeout,
                read_timeout=timeout,
                user_agent_extra="x-client-framework:llama_index",
            )
            if botocore_config is None
            else botocore_config
        )

        self._boto_client_kwargs = {
            "api_version": api_version,
            "use_ssl": use_ssl,
            "verify": verify,
            "endpoint_url": endpoint_url,
        }

        try:
            self._config = (
                Config(
                    retries={"max_attempts": max_retries, "mode": "standard"},
                    connect_timeout=timeout,
                    read_timeout=timeout,
                    user_agent_extra="x-client-framework:llama_index",
                )
                if botocore_config is None
                else botocore_config
            )
            session = boto3.Session(**session_kwargs)
        except ImportError:
            raise ImportError(
                "boto3  package not found, install with pip install boto3"
            )
        session = boto3.Session(**session_kwargs)

        if client is not None:
            self._client = client
        else:
            self._client = session.client(
                "bedrock-agentcore",
                config=self._config,
                **self._boto_client_kwargs,
            )
        super().__init__(self._client, **kwargs)

        self._context = context

    @model_serializer
    def serialize_memory(self) -> Dict[str, Any]:
        # leaving out the two keys since they are causing serialization/deserialization problems
        return {
            "search_msg_limit": self.search_msg_limit,
        }

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "AgentCoreMemory"

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "AgentCoreMemory":
        raise NotImplementedError("Use either from_client or from_config")

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""
        # Get list of events to represent as the chat history. Use this as the query for the memory records. If an input is provided, then also append it to the list of events
        messages = self.list_events(
            memory_id=self._context.memory_id,
            session_id=self._context.session_id,
            actor_id=self._context.actor_id,
        )
        input = convert_messages_to_string(messages, input)

        search_criteria = {"searchQuery": input[:10000]}
        if self._context.memory_strategy_id is not None:
            search_criteria["memoryStrategyId"] = self._context.memory_strategy_id

        memory_records = self.retrieve_memories(
            memory_id=self._context.memory_id,
            namespace=self._context.namespace,
            search_criteria=search_criteria,
        )

        if self.insert_method == InsertMethod.SYSTEM:
            system_message = convert_memory_to_system_message(memory_records)
            # If system message is present
            if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM:
                assert messages[0].content is not None
                system_message = convert_memory_to_system_message(
                    response=memory_records, existing_system_message=messages[0]
                )
            messages.insert(0, system_message)
        elif self.insert_method == InsertMethod.USER:
            # Find the latest user message
            session_idx = next(
                (
                    i
                    for i, msg in enumerate(reversed(messages))
                    if msg.role == MessageRole.USER
                ),
                None,
            )

            memory_content = convert_memory_to_user_message(memory_records)

            if session_idx is not None:
                # Get actual index (since we enumerated in reverse)
                actual_idx = len(messages) - 1 - session_idx
                # Update existing user message since many LLMs have issues with consecutive user msgs
                final_user_content = memory_content + messages[actual_idx].content
                messages[actual_idx] = ChatMessage(
                    content=final_user_content, role=MessageRole.USER
                )
                messages[actual_idx].blocks = [
                    *memory_content.blocks,
                    *messages[actual_idx].blocks,
                ]
            else:
                messages.append(
                    ChatMessage(content=memory_content, role=MessageRole.USER)
                )

        return messages

    def get_all(self) -> List[ChatMessage]:
        """Returns all chat history."""
        return self.list_events(
            memory_id=self._context.memory_id,
            session_id=self._context.session_id,
            actor_id=self._context.actor_id,
        )

    def _add_msgs_to_client_memory(self, messages: List[ChatMessage]) -> None:
        """Add new user and assistant messages to client memory."""
        self.create_event(
            messages=messages,
            memory_id=self._context.memory_id,
            actor_id=self._context.actor_id,
            session_id=self._context.session_id,
        )

    def put(self, message: ChatMessage) -> None:
        """Add message to chat history and client memory."""
        self._add_msgs_to_client_memory([message])

    async def aput(self, message: ChatMessage) -> None:
        """Add a message to the chat store and process waterfall logic if needed."""
        # Add the message to the chat store
        self._add_msgs_to_client_memory([message])

    async def aput_messages(self, messages: List[ChatMessage]) -> None:
        """Add a list of messages to the chat store and process waterfall logic if needed."""
        # Add the messages to the chat store
        self._add_msgs_to_client_memory(messages)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history and add new messages to client memory."""
        initial_chat_len = len(self.get_all())
        # Insert only new chat messages
        self._add_msgs_to_client_memory(messages[initial_chat_len:])

    def reset(self) -> None:
        """Only reset chat history."""
        # Our guidance has been to not delete memory resources in AgentCore on behalf of the customer. If this changes in the future, then we can implement this method.

    def get_context(self) -> AgentCoreMemoryContext:
        return self._context.get_context()
