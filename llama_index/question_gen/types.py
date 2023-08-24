from abc import ABC, abstractmethod
from typing import List, Sequence

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from llama_index.indices.query.schema import QueryBundle
from llama_index.tools.types import ToolMetadata


class SubQuestion(BaseModel):
    sub_question: str
    tool_name: str


class SubQuestionList(BaseModel):
    """A pydantic object wrapping a list of sub-questions.

    This is mostly used to make getting a json schema easier.
    """

    items: List[SubQuestion]


class BaseQuestionGenerator(ABC):
    @abstractmethod
    def generate(self, tools: Sequence[ToolMetadata], query: QueryBundle) -> List[SubQuestion]:
        pass

    @abstractmethod
    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        pass
