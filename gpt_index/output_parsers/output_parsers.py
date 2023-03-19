try:
    from guardrails import Guard, PromptCallable
except ImportError:
    Guard = None
    PromptCallable = None

from typing import Optional

from langchain.llms.base import BaseLLM


def get_prompt_callable(llm: Optional[BaseLLM]) -> PromptCallable:
    if llm is None:
        return None

    return PromptCallable(llm.__call__)


class OutputParser:
    def __init__(self):
        pass

    def parse(self, output: str) -> str:
        raise NotImplementedError

    def format(self, output: str) -> str:
        raise NotImplementedError


class GuardrailsOutputParser:
    def __init__(self, guard: Guard, llm: Optional[BaseLLM] = None):
        self.guard = guard
        self.llm = llm

    @classmethod
    def from_rail(cls, rail: str, llm: Optional[BaseLLM] = None):
        if Guard is None:
            raise ImportError(
                "Guardrails is not installed. Run `pip install guardrails-ai`. "
            )

        return cls(Guard.from_rail(rail), llm=llm)

    @classmethod
    def from_rail_string(cls, rail_string: str, llm: Optional[BaseLLM] = None):
        if Guard is None:
            raise ImportError(
                "Guardrails is not installed. Run `pip install guardrails-ai`. "
            )

        return cls(Guard.from_rail_string(rail_string), llm=llm)

    def parse(
        self,
        output: str,
        llm: Optional[BaseLLM] = None,
        num_reasks: Optional[int] = 1,
        *args,
        **kwargs
    ) -> str:
        llm = llm or self.llm
        prompt_callable = get_prompt_callable(llm)

        return self.guard.parse(
            output,
            prompt_callable=prompt_callable,
            num_reasks=num_reasks,
            *args,
            **kwargs
        )

    def format(self, query: str) -> str:
        # Add format instructions here.
        pass
