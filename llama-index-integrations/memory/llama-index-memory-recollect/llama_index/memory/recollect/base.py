from __future__ import annotations

from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import BaseMemory
from llama_index.core.memory import Memory as LlamaIndexMemory
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from llama_index.memory.recollect.utils import (
    convert_chat_history_to_dict,
    convert_memory_to_system_message,
    convert_messages_to_string,
)
from recollect.config import RecollectConfig
from recollect.memory import Memory


class RecollectContext(BaseModel):
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None

    def validate_provided_context(self) -> None:
        if all(v is None for v in (self.user_id, self.agent_id, self.run_id)):
            raise ValueError("Recollect context requires at least one of user_id, agent_id, run_id")

    def build_filter(self) -> dict[str, str]:
        flt: dict[str, str] = {}
        if self.user_id is not None:
            flt["user_id"] = self.user_id
        if self.agent_id is not None:
            flt["agent_id"] = self.agent_id
        if self.run_id is not None:
            flt["run_id"] = self.run_id
        return flt

    def scope_kwargs(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class RecollectMemory(BaseMemory):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_memory: LlamaIndexMemory = Field(description="Chat history buffer")
    context: RecollectContext = Field(default_factory=RecollectContext)
    search_msg_limit: int = Field(default=5)

    _client: Memory | None = PrivateAttr(default=None)

    def __init__(
        self,
        context: RecollectContext | dict[str, Any] | None = None,
        client: Memory | None = None,
        **kwargs: Any,
    ) -> None:
        if "primary_memory" not in kwargs:
            kwargs["primary_memory"] = LlamaIndexMemory.from_defaults()
        super().__init__(**kwargs)
        self._client = client
        if context is not None:
            ctx = RecollectContext.model_validate(context)
            ctx.validate_provided_context()
            self.context = ctx

    @classmethod
    def class_name(cls) -> str:
        return "RecollectMemory"

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> RecollectMemory:
        raise NotImplementedError("Use RecollectMemory.from_config()")

    @classmethod
    def from_config(
        cls,
        context: dict[str, Any],
        config: RecollectConfig | dict[str, Any] | None = None,
        search_msg_limit: int = 5,
        **kwargs: Any,
    ) -> RecollectMemory:
        ctx = RecollectContext.model_validate(context)
        ctx.validate_provided_context()
        if isinstance(config, dict):
            cfg = RecollectConfig.model_validate(config)
        elif config is None:
            cfg = RecollectConfig.local_dev()
        else:
            cfg = config
        client = Memory(cfg)
        return cls(
            primary_memory=LlamaIndexMemory.from_defaults(),
            context=ctx,
            client=client,
            search_msg_limit=search_msg_limit,
            **kwargs,
        )

    def get(self, input: str | None = None, **kwargs: Any) -> list[ChatMessage]:
        assert self._client is not None
        messages = self.primary_memory.get(input=input, **kwargs)
        query = convert_messages_to_string(messages, input, limit=self.search_msg_limit)
        result = self._client.search(query, filters=self.context.build_filter())
        search_results = result.get("results", [])
        system_message = convert_memory_to_system_message(search_results)
        if messages and messages[0].role == MessageRole.SYSTEM:
            system_message = convert_memory_to_system_message(
                search_results, existing_system_message=messages[0]
            )
        messages.insert(0, system_message)
        return messages

    def get_all(self) -> list[ChatMessage]:
        return self.primary_memory.get_all()

    def _add_msgs_to_client_memory(self, messages: list[ChatMessage]) -> None:
        assert self._client is not None
        self._client.add(
            convert_chat_history_to_dict(messages),
            **self.context.scope_kwargs(),
        )

    def put(self, message: ChatMessage) -> None:
        self._add_msgs_to_client_memory([message])
        self.primary_memory.put(message)

    def set(self, messages: list[ChatMessage]) -> None:
        initial_len = len(self.primary_memory.get_all())
        self._add_msgs_to_client_memory(messages[initial_len:])
        self.primary_memory.set(messages)

    def reset(self) -> None:
        self.primary_memory.reset()