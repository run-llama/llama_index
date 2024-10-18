
from typing import Any, Dict, List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.memory.mem0.base import BaseMem0, Mem0Context
from llama_index.memory.mem0.utils import convert_memory_to_system_message, format_memory_json
from pydantic import ValidationError
from mem0 import Memory, MemoryClient

class Mem0ChatEngineMemory(BaseMem0):
    buffer_user_message: Optional[ChatMessage] = None
    
    #TODO: Implement better 
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @staticmethod
    def class_name() -> str:
        """Get class name."""
        return "Mem0ChatEngineMemory"
    
    #TODO: Implement better
    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "Mem0ChatEngineMemory":
        return cls(**kwargs)
    
    @classmethod
    def from_client(
        cls,
        context_dict: Dict[str, Any],
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None
    ) :
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
        return cls(client=client, context=context)
    
    @classmethod
    def from_config(
        cls, 
        context_dict: Dict[str, Any], 
        confif_dict: Dict[str, Any]
    ):
        try:
            context = Mem0Context(**context_dict)
        except Exception as e:
            raise ValidationError(f"Context validation error: {e}")

        client = Memory.from_config(config_dict=confif_dict)
        return cls(context=context, client=client)

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        #TODO: Remove the repetitive memory
        search_query = str(self.buffer_user_message.content)
        if input is not None:
            search_query = input
        responses = self.client.search(
            query=search_query,
            **self.context.get_context(),
        )
        if isinstance(self.client, Memory):
            responses = responses['results']
        response_messages = [convert_memory_to_system_message(responses)]
        # chat engine not sending latest message
        response_messages.append(self.buffer_user_message)
        return response_messages

    def put(self, message: ChatMessage) -> None:
        #TODO: Remove the repetitive memory
        if message.role == MessageRole.USER:
            self.buffer_user_message = message
            add_msg = str(message.content) if not isinstance(message.content, str) else message.content
            self.client.add(
                messages=add_msg,
                **self.context.get_context()
            )
    
    #TODO: improve
    def get_all(self) -> List[ChatMessage]:
        responses = self.client.get_all(**self.context.get_context())
        messages = [ChatMessage(
            content = format_memory_json(response))
            for response in responses]
        return messages
    
    def set(self, messages: List[ChatMessage]) -> None:
        user_messages = [
            {"role": "user", "content": msg.content}
            for msg in messages
            if msg.role == MessageRole.USER
        ]
        if user_messages:
            self.client.add(messages=user_messages, **self.context.get_context())
    
    def reset(self) -> None:
        self.client.reset()
    
    # Dummy tokenizer for legacy chat_engines 
    def tokenizer_fn(*args, **kwargs):
        return ""