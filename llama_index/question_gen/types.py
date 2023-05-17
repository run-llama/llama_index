


from typing import List, Sequence

from pydantic import BaseModel

from llama_index.indices.query.schema import QueryBundle
from llama_index.tools.types import ToolMetadata


class SubQuestion(BaseModel):
    sub_question: str
    tool_name: str


class BaseQuestionGenerator:
    def generate(self, tools: Sequence[ToolMetadata], query: QueryBundle) -> List[SubQuestion]:
        return []

    async def agenerate(self, tools: Sequence[ToolMetadata], query: QueryBundle) -> List[SubQuestion]:
        return []
