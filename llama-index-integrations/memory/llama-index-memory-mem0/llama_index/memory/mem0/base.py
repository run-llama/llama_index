from typing import Any, Dict, List, Optional, Union
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseMemory
from mem0 import MemoryClient, Memory
from pydantic import BaseModel, ValidationError, model_validator

DEFAULT_INTRO_PREFERENCES = "Below are a set of relevant preferences retrieved from potentially several memory sources:"
DEFAULT_OUTRO_PREFERENCES = "This is the end of the retrieved preferences."

class Mem0Context(BaseModel):
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    def get_context(self) -> Dict[str, Optional[str]]:
        return {
            key: value for key, value in self.__dict__.items()
            if value is not None
        }
    
    @model_validator(mode='after')
    def check_at_least_one_id(cls, values):
        if not any([values.user_id, values.agent_id, values.run_id]):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be provided!")
        return values


class Mem0(BaseMemory):
    class Config:
        arbitrary_types_allowed = True 

    context: Optional[Mem0Context] = None
    client: Optional[Union[MemoryClient, Memory]] = None
    buffer_user_message: Optional[ChatMessage] = None
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def _format_memory_json(self, memory_json: Dict[str, Any]) -> List[str]:
        categories = memory_json.get('categories')
        memory = memory_json.get('memory', '')
        if categories is not None:
            categories_str = ', '.join(categories)
            return f"[{categories_str}] : {memory}"
        return f"{memory}"
    
    def _convert_memory_to_system_message(self, response: List[Dict[str, Any]]) -> ChatMessage:
        memories = [self._format_memory_json(memory_json) for memory_json in response]
        formatted_messages = "\n\n" + DEFAULT_INTRO_PREFERENCES + "\n"
        for memory in memories:
            formatted_messages += (
                f"\n {memory} \n\n"
            )
        formatted_messages += DEFAULT_OUTRO_PREFERENCES
        return ChatMessage(content = formatted_messages, role=MessageRole.SYSTEM)
    
    @staticmethod
    def class_name() -> str:
        """Get class name."""
        return "Mem0"
    
    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "Mem0":
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

    def set_context(self, context_dict: Dict[str, str]) -> None:
        self.context = Mem0Context(**context_dict)

    def get_context(self) -> Dict[str, Optional[str]]:
        return self.context.get_context()

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
        response_messages = [self._convert_memory_to_system_message(responses)]
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
            content = self._format_memory_json(response))
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


