from abc import abstractmethod
from typing import List, Sequence

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.prompts.mixin import PromptMixin, PromptMixinType
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata


class SubQuestion(BaseModel):
    sub_question: str
    tool_name: str


class SubQuestionList(BaseModel):
    """
    A pydantic object wrapping a list of sub-questions.

    This is mostly used to make getting a json schema easier.
    """

    items: List[SubQuestion]


class BaseQuestionGenerator(PromptMixin, DispatcherSpanMixin):
    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    @abstractmethod
    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        pass

    @abstractmethod
    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        pass
