from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, Field

from llama_index.bridge.langchain import BasePromptTemplate
from llama_index.llms.base import LLM


class PromptSelector(BaseModel):
    default_prompt: BasePromptTemplate
    conditionals: List[Tuple[Callable[[LLM], bool], BasePromptTemplate]] = Field(
        default_factory=list
    )

    def select(self, llm: Optional[LLM] = None) -> BasePromptTemplate:
        if llm is None:
            return self.default_prompt

        for condition, prompt in self.conditionals:
            if condition(llm):
                return prompt
        return self.default_prompt


def is_chat_model(llm: LLM) -> bool:
    return llm.metadata.is_chat_model
