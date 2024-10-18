from typing import Dict, Optional, Union
from llama_index.core.memory.types import BaseMemory
from mem0 import MemoryClient, Memory
from pydantic import BaseModel, model_validator

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


class BaseMem0(BaseMemory):
    class Config:
        arbitrary_types_allowed = True 

    context: Optional[Mem0Context] = None
    client: Optional[Union[MemoryClient, Memory]] = None

    def set_context(self, context_dict: Dict[str, str]) -> None:
        self.context = Mem0Context(**context_dict)

    def get_context(self) -> Dict[str, Optional[str]]:
        return self.context.get_context()