
from typing import Any, Dict, List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseMemory
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.memory.mem0.base import BaseMem0, Mem0Context
from llama_index.memory.mem0.utils import convert_memory_to_system_message
from pydantic import Field, SerializeAsAny, ValidationError
from mem0 import Memory, MemoryClient


class Mem0FunctionalAgentMemory(BaseMem0):
    
    history: Dict[str, Any] = {}
    primary_memory: SerializeAsAny[BaseMemory] = Field(
        description="Primary memory source for chat agent.",
    )

    #TODO: Implement better
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "Mem0FunctionalAgentMemory":
        return cls(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "Mem0ComposableMemory"
    
    @classmethod
    def from_client(
        cls,
        context_dict: Dict[str, Any],
        primary_memory: Optional[BaseMemory] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        **kwargs: Any,
    ) :
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")
        
        primary_memory = primary_memory or ChatMemoryBuffer.from_defaults()

        try:
            context = Mem0Context(**context_dict)
        except Exception as e:
            raise ValidationError(f"Context validation error: {e}")
               
        client = MemoryClient(
            api_key=api_key,
            host=host,
            organization=organization,
            project=project
        )
        return cls(primary_memory=primary_memory, client=client, context=context)
    
    @classmethod
    def from_config(
        cls,
        context_dict: Dict[str, Any], 
        confif_dict: Dict[str, Any],
        primary_memory: Optional[BaseMemory] = None,
        **kwargs: Any,
    ):
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        primary_memory = primary_memory or ChatMemoryBuffer.from_defaults()

        try:
            context = Mem0Context(**context_dict)
        except Exception as e:
            raise ValidationError(f"Context validation error: {e}")

        client = Memory.from_config(config_dict=confif_dict)
        return cls(primary_memory=primary_memory, context=context, client=client)
    

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        if input is None:
            raise ValueError("input string not found.")
        return self._compose_memory_messages(input, **kwargs)
    
    def _compose_memory_messages(
        self, input: Optional[str] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        messages = self.primary_memory.get(input=input, **kwargs)
        #TODO: Add keywords args according to client
        responses = self.client.search(
            query=input,
            **self.context.get_context(),
        )
        if isinstance(self.client, Memory):
            responses = responses['results']
        system_message = convert_memory_to_system_message(responses)
        if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM:
            assert messages[0].content is not None
            system_message = convert_memory_to_system_message(response=responses, existing_system_message=messages[0])
        messages.insert(0, system_message)

        return messages
    
    # for functional agents only
    def get_all(self) -> List[ChatMessage]:
        return self.primary_memory.get_all()
    
    def put(self, message: ChatMessage) -> None:
        # only puts in mem0
        if message.role == MessageRole.USER:
            msg_str = str(message.content)
            if msg_str not in self.history:
                response = self.client.add(
                    messages=msg_str,
                    **self.context.get_context()
                )
                self.history[msg_str] = response

    def set(self, messages: List[ChatMessage]) -> None:
        self.primary_memory.set(messages)
        for message in messages:
            self.put(message)

    def reset(self) -> None:
        self.primary_memory.reset()
        self.client.reset() 
