from typing import Dict, List, Optional, Union, Any
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.memory.mem0.utils import convert_memory_to_system_message
from mem0 import MemoryClient, Memory
from pydantic import BaseModel, Field, ValidationError, model_validator, SerializeAsAny
from llama_index.core.base.llms.types import ChatMessage, MessageRole

class BaseMem0(BaseMemory):
    """Base class for Mem0"""
    class Config:
        arbitrary_types_allowed = True 

    client: Optional[Union[MemoryClient, Memory]] = None
    
    #TODO: Return type
    def add(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> Optional[Any]:
        response = self.client.add(messages=messages, **kwargs)
        return response

    #TODO: Return type
    def search(self, query: str, **kwargs) -> Optional[Any]:
        response = self.client.search(query=query, **kwargs)
        return response
    
    #TODO: Add more apis from client

class Mem0Context(BaseModel):
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    @model_validator(mode="after")
    def check_at_least_one_assigned(cls, values):
        if not any(getattr(values, field) for field in ['user_id', 'agent_id', 'run_id']):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be assigned.")
        return values
    
    def get_context(self) -> Dict[str, Optional[str]]:
        return {
            key: value for key, value in self.__dict__.items()
            if value is not None
        }
    
    class Config:
        validate_assignment = True

class Mem0Composable(BaseMem0):
    #TODO: Make it private variable
    chat_history: SerializeAsAny[BaseMemory] = Field(
        description="Primary memory source for chat agent.",
    )
    #TODO: Make it private variable
    mem0_history: Dict[str, Any] = {}
    context: Optional[Mem0Context] = None

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "Mem0Composable"
    
    #TODO: Not functional yet. 
    @classmethod 
    def from_defaults(cls, **kwargs: Any) -> "Mem0Composable":
        raise NotImplementedError("Use either from_client or from_config")
    
    @classmethod
    def from_client(
        cls,
        context_dict: Dict[str, Any],
        chat_history: Optional[BaseMemory] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        **kwargs: Any,
    ) :
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")
        
        chat_history = chat_history or ChatMemoryBuffer.from_defaults()

        try:
            context = Mem0Context(**context_dict)
        except ValidationError as e:
            raise ValidationError(f"Context validation error: {e}")

        client = MemoryClient(
            api_key=api_key,
            host=host,
            organization=organization,
            project=project
        )
        return cls(chat_history=chat_history, client=client, context=context)
    
    @classmethod
    def from_config(
        cls,
        context_dict: Dict[str, Any], 
        confif_dict: Dict[str, Any],
        chat_history: Optional[BaseMemory] = None,
        **kwargs: Any,
    ):
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        chat_history = chat_history or ChatMemoryBuffer.from_defaults()

        try:
            context = Mem0Context(**context_dict)
        except Exception as e:
            raise ValidationError(f"Context validation error: {e}")

        client = Memory.from_config(config_dict=confif_dict)
        return cls(chat_history=chat_history, context=context, client=client)
    
    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        messages = self.chat_history.get(input=input, **kwargs)
        if input is None:
            # Iterate through messages from last to first
            for message in reversed(messages):
                if message.role == MessageRole.USER:
                    most_recent_user_message = message
                    break
            else:
                # If no user message is found, raise an exception
                raise ValueError("No user message found in chat history")
            input = str(most_recent_user_message.content)

        #TODO: Add support for more kwargs, for api and oss
        search_results = self.search(query=input, **self.context.get_context())
        if isinstance(self.client, Memory):
            search_results = search_results['results']
        system_message = convert_memory_to_system_message(search_results)
        
        #TODO: What if users provide system_message or prefix_message, or system_message in chat_history becaomes old.
        if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM:
            assert messages[0].content is not None
            system_message = convert_memory_to_system_message(response=search_results, existing_system_message=messages[0])
        messages.insert(0, system_message)
        return messages

    def get_all(self) -> List[ChatMessage]:
        """Returns all chat history."""
        return self.chat_history.get_all()

    def _add_to_memory(self, message: ChatMessage) -> None:
        """Only add new user message to client memory."""
        if message.role == MessageRole.USER:
            msg_str = str(message.content)
            if msg_str not in self.mem0_history:
                #TODO: Implement for more kwargs
                response = self.client.add(
                    messages=msg_str,
                    **self.context.get_context()
                )
                self.mem0_history[msg_str] = response

    def put(self, message: ChatMessage) -> None:
        """Add message to chat history. Add new user message to client memory."""
        self.chat_history.put(message)
        self._add_to_memory(message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history. Add new user message to client memory."""
        self.chat_history.set(messages)
        for message in messages:
            self._add_to_memory(message)

    def reset(self) -> None:
        """Only reset chat history"""
        #TODO: Not resetting client memory, since it is not context specific.
        self.chat_history.reset()




    

    
    