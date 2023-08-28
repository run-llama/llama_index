from typing import Callable, List

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from llama_index.tools.types import BaseTool


class Task(BaseModel):
    message: str
    expected_response: str
    tools: List[BaseTool]
    eval_fn: Callable[[str, str], bool]

    class Config:
        arbitrary_types_allowed = True
