from typing import Dict, List, Optional, Union, Any

from llama_index.core.memory import BaseMemory, Memory as LlamaIndexMemory
from llama_index.memory.mem0.utils import (
    convert_memory_to_system_message,
    convert_chat_history_to_dict,
    convert_messages_to_string,
)
from mem0 import MemoryClient, Memory
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
    SerializeAsAny,
    PrivateAttr,
    ConfigDict,
    model_serializer,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole


class BaseMem0(BaseMemory):
    """Base class for Mem0."""

    _client: Optional[Union[MemoryClient, Memory]] = PrivateAttr()

    def __init__(
        self, client: Optional[Union[MemoryClient, Memory]] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if client is not None:
            self._client = client

    def add(
        self, messages: Union[str, List[Dict[str, str]]], **kwargs
    ) -> Optional[Dict[str, Any]]:
        if self._client is None:
            raise ValueError("Client is not initialized")
        if not messages:
            return None
        return self._client.add(messages=messages, **kwargs)

    def search(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        if self._client is None:
            raise ValueError("Client is not initialized")
        return self._client.search(query=query, **kwargs)


class Mem0Context(BaseModel):
    """Context identifiers for Mem0 memory."""

    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

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
        """Return non-null context values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Mem0Memory(BaseMem0):
    """Mem0-backed memory integration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_memory: SerializeAsAny[LlamaIndexMemory] = Field(
        description="Primary memory source for chat agent."
    )
    context: Optional[Mem0Context] = None
    search_msg_limit: int = Field(
        default=5,
        description="Limit of chat history messages used for search context",
    )

    def __init__(self, context: Optional[Mem0Context] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if context is not None:
            self.context = context

    @model_serializer
    def serialize_memory(self) -> Dict[str, Any]:
        """Serialize memory state."""
        # leaving out keys that cause serialization/deserialization issues
        return {
            "primary_memory": self.primary_memory.model_dump(
                exclude={"memory_blocks_template", "insert_method"}
            ),
            "search_msg_limit": self.search_msg_limit,
            "context": self.context.model_dump() if self.context else None,
        }

    @classmethod
    def class_name(cls) -> str:
        return "Mem0Memory"

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "Mem0Memory":
        raise NotImplementedError("Use either from_client or from_config")

    @classmethod
    def from_client(
        cls,
        context: Dict[str, Any],
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        search_msg_limit: int = 5,
        **kwargs: Any,
    ):
        primary_memory = LlamaIndexMemory.from_defaults()

        try:
            context = Mem0Context(**context)
        except ValidationError as e:
            raise ValidationError(f"Context validation error: {e}")

        client = MemoryClient(
            api_key=api_key, host=host, org_id=org_id, project_id=project_id
        )

        return cls(
            primary_memory=primary_memory,
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
        primary_memory = LlamaIndexMemory.from_defaults()

        try:
            context = Mem0Context(**context)
        except ValidationError as e:
            raise ValidationError(f"Context validation error: {e}")

        client = Memory.from_config(config_dict=config)

        return cls(
            primary_memory=primary_memory,
            context=context,
            client=client,
            search_msg_limit=search_msg_limit,
        )

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history with Mem0-enriched system message."""
        messages = self.primary_memory.get(input=input, **kwargs)
        query = convert_messages_to_string(
            messages, input, limit=self.search_msg_limit
        )

        context_dict = self.context.get_context()
        search_results: List[Dict[str, Any]] = []

        if "user_id" in context_dict and "agent_id" in context_dict:
            user_results = self.search(query=query, user_id=context_dict["user_id"])
            agent_results = self.search(query=query, agent_id=context_dict["agent_id"])

            if isinstance(self._client, Memory) and self._client.api_version == "v1.1":
                user_results = user_results.get("results", [])
                agent_results = agent_results.get("results", [])

            # Deduplicate merged results
            seen = set()
            for item in (user_results or []) + (agent_results or []):
                content = item.get("content")
                if content and content not in seen:
                    seen.add(content)
                    search_results.append(item)
        else:
            search_results = self.search(query=query, **context_dict) or []
            if isinstance(self._client, Memory) and self._client.api_version == "v1.1":
                search_results = search_results.get("results", [])

        system_message = convert_memory_to_system_message(search_results)

        if messages and messages[0].role == MessageRole.SYSTEM:
            system_message = convert_memory_to_system_message(
                response=search_results, existing_system_message=messages[0]
            )

        messages.insert(0, system_message)
        return messages

    def get_all(self) -> List[ChatMessage]:
        """Return full chat history."""
        return self.primary_memory.get_all()

    def _add_msgs_to_client_memory(self, messages: List[ChatMessage]) -> None:
        """Add messages to Mem0 client."""
        self.add(
            messages=convert_chat_history_to_dict(messages),
            **self.context.get_context(),
        )

    def put(self, message: ChatMessage) -> None:
        """Add message to memory."""
        self._add_msgs_to_client_memory([message])
        self.primary_memory.put(message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history and sync new messages to Mem0."""
        initial_len = len(self.primary_memory.get_all())
        self._add_msgs_to_client_memory(messages[initial_len:])
        self.primary_memory.set(messages)

    def reset(self) -> None:
        """Reset only local chat history."""
        self.primary_memory.reset()
