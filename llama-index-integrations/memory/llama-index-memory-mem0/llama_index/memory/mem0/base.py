from typing import Any, Dict, List, Optional, Union, cast

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializeAsAny,
    ValidationError,
    model_serializer,
)
from llama_index.core.memory import BaseMemory
from llama_index.core.memory import Memory as LlamaIndexMemory
from llama_index.memory.mem0.utils import (
    convert_chat_history_to_dict,
    convert_memory_to_system_message,
    convert_messages_to_string,
)
from typing_extensions import NotRequired, TypedDict

from mem0 import Memory, MemoryClient


class Mem0AddResult(TypedDict):
    results: list[Any]
    relations: NotRequired[Any]


class Mem0SearchResult(TypedDict):
    results: Any


class BaseMem0(BaseMemory):
    """Base class for Mem0."""

    _client: Optional[Union[MemoryClient, Memory]] = PrivateAttr()

    def __init__(
        self, client: Optional[Union[MemoryClient, Memory]] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._client = client

    def add(
        self,
        messages: Union[str, Dict[str, str], List[Dict[str, str]]],
        **kwargs: Any,
    ) -> Optional[Union[Mem0AddResult, Dict[str, Any]]]:
        assert self._client is not None, (
            "Client should be not-null when performing memory operations"
        )
        if not messages:
            return None
        return self._client.add(messages=messages, **kwargs)

    def search(self, query: str, **kwargs) -> Mem0SearchResult:
        assert self._client is not None, (
            "Client should be not-null when performing memory operations"
        )
        result = self._client.search(query=query, **kwargs)
        if isinstance(result, list):
            search_result: Mem0SearchResult = {"results": result}
            return search_result
        # client should return results in the form of {'results': ...}
        return cast(Mem0SearchResult, result)


class Mem0Context(BaseModel):
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    def validate_provided_context(self) -> None:
        if all(el is None for el in [self.user_id, self.agent_id, self.run_id]):
            raise ValidationError(
                "When providing a non-default context, you should have at least one not-null field"
            )

    def build_filter(self) -> dict[str, Any]:
        flt = {"OR": []}
        if self.user_id is not None:
            flt["OR"].append({"user_id": self.user_id})
        if self.run_id is not None:
            flt["OR"].append({"run_id": self.run_id})
        if self.agent_id is not None:
            flt["OR"].append({"agent_id": self.agent_id})
        return flt

    @model_serializer
    def serialize_with_omitempty(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        if self.user_id is not None:
            context.update({"user_id": self.user_id})
        if self.run_id is not None:
            context.update({"run_id": self.run_id})
        if self.agent_id is not None:
            context.update({"agent_id": self.agent_id})
        return context


class Mem0Memory(BaseMem0):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    primary_memory: SerializeAsAny[LlamaIndexMemory] = Field(
        description="Primary memory source for chat agent."
    )
    context: Mem0Context = Mem0Context()
    search_msg_limit: int = Field(
        default=5,
        description="Limit of chat history messages to use for context in search API",
    )

    def __init__(
        self, context: Optional[Union[Mem0Context, dict[str, Any]]] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if context is not None:
            if isinstance(context, dict):
                context = Mem0Context.model_validate(context)
            context.validate_provided_context()
            self.context = context

    @model_serializer
    def serialize_memory(self) -> Dict[str, Any]:
        # leaving out the two keys since they are causing serialization/deserialization problems
        return {
            "primary_memory": self.primary_memory.model_dump(
                exclude={
                    "memory_blocks_template",
                    "insert_method",
                }
            ),
            "search_msg_limit": self.search_msg_limit,
            "context": self.context.model_dump(),
        }

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
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
            mem0_ctx = Mem0Context.model_validate(context)
        except ValidationError as e:
            raise ValidationError(f"Context validation error: {e}")

        client = MemoryClient(
            api_key=api_key, host=host, org_id=org_id, project_id=project_id
        )
        return cls(
            primary_memory=primary_memory,
            context=mem0_ctx,
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
            mem0_ctx = Mem0Context(**context)
        except Exception as e:
            raise ValidationError(f"Context validation error: {e}")

        client = Memory.from_config(config_dict=config)
        return cls(
            primary_memory=primary_memory,
            context=mem0_ctx,
            client=client,
            search_msg_limit=search_msg_limit,
        )

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history. With memory system message."""
        messages = self.primary_memory.get(input=input, **kwargs)
        input = convert_messages_to_string(messages, input, limit=self.search_msg_limit)
        ctx = self.context.model_dump()
        if len(ctx) > 1:
            flt = self.context.build_filter()
            result = self.search(query=input, filters=flt)
        else:
            result = self.search(query=input, **ctx)

        search_results = result.get("results", [])

        system_message = convert_memory_to_system_message(search_results)

        # If system message is present
        if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM:
            assert messages[0].content is not None
            system_message = convert_memory_to_system_message(
                response=search_results, existing_system_message=messages[0]
            )
        messages.insert(0, system_message)
        return messages

    def get_all(self) -> List[ChatMessage]:
        """Returns all chat history."""
        return self.primary_memory.get_all()

    def _add_msgs_to_client_memory(self, messages: List[ChatMessage]) -> None:
        """Add new user and assistant messages to client memory."""
        self.add(
            messages=convert_chat_history_to_dict(messages),
            **self.context.model_dump(),
        )

    def put(self, message: ChatMessage) -> None:
        """Add message to chat history and client memory."""
        self._add_msgs_to_client_memory([message])
        self.primary_memory.put(message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history and add new messages to client memory."""
        initial_chat_len = len(self.primary_memory.get_all())
        # Insert only new chat messages
        self._add_msgs_to_client_memory(messages[initial_chat_len:])
        self.primary_memory.set(messages)

    def reset(self) -> None:
        """Only reset chat history."""
        self.primary_memory.reset()
