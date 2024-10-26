from typing import Dict, List, Optional, Union, Any
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.memory.mem0.utils import (
    convert_memory_to_system_message,
    convert_chat_history_to_dict,
    convert_messages_to_string,
)
from mem0 import MemoryClient, Memory
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
    SerializeAsAny,
    PrivateAttr,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole


class BaseMem0(BaseMemory):
    """Base class for Mem0."""

    _client: Optional[Union[MemoryClient, Memory]] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if "client" in data:
            object.__setattr__(self, "_client", data["client"])

    # TODO: Return type
    def add(
        self, messages: Union[str, List[Dict[str, str]]], **kwargs
    ) -> Optional[Any]:
        if self._client is None:
            raise ValueError("Client is not initialized")
        return self._client.add(messages=messages, **kwargs)

    # TODO: Return type
    def search(self, query: str, **kwargs) -> Optional[Any]:
        if self._client is None:
            raise ValueError("Client is not initialized")
        return self._client.search(query=query, **kwargs)

    # TODO: Add more apis from client


class Mem0Context(BaseModel):
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_at_least_one_assigned(cls, values):
        if not any(
            getattr(values, field) for field in ["user_id", "agent_id", "run_id"]
        ):
            raise ValueError(
                "At least one of 'user_id', 'agent_id', or 'run_id' must be assigned."
            )
        return values

    def get_context(self) -> Dict[str, Optional[str]]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


class Mem0Memory(BaseMem0):
    chat_history: SerializeAsAny[BaseMemory] = Field(
        description="Primary memory source for chat agent."
    )
    _context: Optional[Mem0Context] = PrivateAttr(default=None)
    search_msg_limit: int = Field(
        default=5,
        description="Limit of chat history messages to use for context in search API",
    )

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if "context" in data:
            object.__setattr__(self, "_context", data["context"])

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "Mem0Memory"

    # TODO: Not functional yet.
    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "Mem0Memory":
        raise NotImplementedError("Use either from_client or from_config")

    @classmethod
    def from_client(
        cls,
        context: Dict[str, Any],
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        search_msg_limit: int = 5,
        **kwargs: Any,
    ):
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        chat_history = ChatMemoryBuffer.from_defaults()

        try:
            context = Mem0Context(**context)
        except ValidationError as e:
            raise ValidationError(f"Context validation error: {e}")

        client = MemoryClient(
            api_key=api_key, host=host, organization=organization, project=project
        )
        return cls(
            chat_history=chat_history,
            context=context,
            client=client,
            search_msg_limit=search_msg_limit,
        )

    @classmethod
    def from_config(
        cls,
        context: Dict[str, Any],
        config: Dict[str, Any],
        search_msg_limit: int = 5,
        **kwargs: Any,
    ):
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        chat_history = ChatMemoryBuffer.from_defaults()

        try:
            context = Mem0Context(**context)
        except Exception as e:
            raise ValidationError(f"Context validation error: {e}")

        client = Memory.from_config(config_dict=config)
        return cls(
            chat_history=chat_history,
            context=context,
            client=client,
            search_msg_limit=search_msg_limit,
        )

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history. With memory system message."""
        messages = self.chat_history.get(input=input, **kwargs)
        input = convert_messages_to_string(messages, input, limit=self.search_msg_limit)

        # TODO: Add support for more kwargs, for api and oss
        search_results = self.search(query=input, **self._context.get_context())
        if isinstance(self._client, Memory):
            search_results = search_results["results"]
        system_message = convert_memory_to_system_message(search_results)

        # If system message is present
        if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM:
            # TODO: What if users provide system_message or prefix_message, or system_message in chat_history becaomes old enough?
            assert messages[0].content is not None
            system_message = convert_memory_to_system_message(
                response=search_results, existing_system_message=messages[0]
            )
        messages.insert(0, system_message)
        return messages

    def get_all(self) -> List[ChatMessage]:
        """Returns all chat history."""
        return self.chat_history.get_all()

    def _add_msgs_to_client_memory(self, messages: List[ChatMessage]) -> None:
        """Add new user and assistant messages to client memory."""
        self.add(
            messages=convert_chat_history_to_dict(messages),
            **self._context.get_context(),
        )

    def put(self, message: ChatMessage) -> None:
        """Add message to chat history and client memory."""
        self._add_msgs_to_client_memory([message])
        self.chat_history.put(message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history and add new messages to client memory."""
        initial_chat_len = len(self.chat_history.get_all())
        # Insert only new chat messages
        self._add_msgs_to_client_memory(messages[initial_chat_len:])
        self.chat_history.set(messages)

    def reset(self) -> None:
        """Only reset chat history."""
        # TODO: Context specific reset is missing in client.
        self.chat_history.reset()
