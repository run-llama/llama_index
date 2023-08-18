"""Prompts."""


from copy import deepcopy
from typing import Any, Callable, List, Optional, Protocol, Tuple

from pydantic import BaseModel, Field

from llama_index.llms.base import LLM, ChatMessage
from llama_index.llms.generic_utils import messages_to_prompt, prompt_to_messages
from llama_index.prompts import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.utils import get_template_vars


class PromptTemplate(Protocol):
    @property
    def metadata(self) -> dict:
        ...

    def partial_format(self, **kwargs: Any) -> "PromptTemplate":
        ...

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        ...

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        ...


class ChatPromptTemplate:
    def __init__(
        self,
        message_templates: List[ChatMessage],
        prompt_type: str = PromptType.CUSTOM,
        **kwargs: Any,
    ):
        self._message_templates = message_templates
        self._kwargs = kwargs

        self._metadata = {
            "prompt_type": prompt_type,
        }

    @property
    def metadata(self) -> dict:
        return self._metadata

    def partial_format(self, **kwargs: Any) -> "ChatPromptTemplate":
        prompt = deepcopy(self)
        prompt._kwargs.update(kwargs)
        return prompt

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        del llm  # unused
        messages = self.format_messages(**kwargs)
        prompt = messages_to_prompt(messages)
        return prompt

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        del llm  # unused
        """Format the prompt into a list of chat messages."""
        all_kwargs = {
            **self._kwargs,
            **kwargs,
        }

        messages = []
        for message_template in self._message_templates:
            template_vars = get_template_vars(message_template.content)
            relevant_kwargs = {
                k: v for k, v in all_kwargs.items() if k in template_vars
            }
            formatted_content = message_template.content.format(**relevant_kwargs)

            message = message_template.copy()
            message.content = formatted_content
            messages.append(message)

        return messages


class PromptTemplate:
    def __init__(
        self,
        template: Optional[str] = None,
        prompt_type: str = PromptType.CUSTOM,
        **kwargs: Any,
    ) -> None:
        self._template = template
        self._kwargs = kwargs

        self._metadata = {
            "prompt_type": prompt_type,
        }

    @property
    def metadata(self) -> dict:
        return self._metadata

    def partial_format(self, **kwargs: Any) -> "PromptTemplate":
        """Partially format the prompt."""
        prompt = deepcopy(self)
        prompt._kwargs.update(kwargs)
        return prompt

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        del llm  # unused
        all_kwargs = {
            **self._kwargs,
            **kwargs,
        }
        return self._template.format(**all_kwargs)

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        del llm  # unused
        prompt = self.format(**kwargs)
        return prompt_to_messages(prompt)


class SelectorPromptTemplate(BaseModel):
    default_prompt: PromptTemplate
    conditionals: List[Tuple[Callable[[LLM], bool], PromptTemplate]] = Field(
        default_factory=list
    )

    @property
    def metadata(self) -> dict:
        return self.default_prompt.metadata

    def _select(self, llm: Optional[LLM] = None) -> PromptTemplate:
        if llm is None:
            return self.default_prompt

        for condition, prompt in self.conditionals:
            if condition(llm):
                return prompt
        return self.default_prompt

    def partial_format(self, **kwargs: Any) -> "SelectorPromptTemplate":
        default_prompt = self.default_prompt.partial_format(**kwargs)
        conditionals = [
            (condition, prompt.partial_format(**kwargs))
            for condition, prompt in self.conditionals
        ]
        return SelectorPromptTemplate(
            default_prompt=default_prompt, conditionals=conditionals
        )

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        prompt = self._select(llm=llm)
        return prompt.format(**kwargs)

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        prompt = self._select(llm=llm)
        return prompt.format(**kwargs)


def is_chat_model(llm: LLM) -> bool:
    return llm.metadata.is_chat_model


# NOTE: only for backwards compatibility
Prompt = PromptTemplate
