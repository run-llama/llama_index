"""Prompts."""


from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from llama_index.bridge.langchain import BasePromptTemplate as LangchainTemplate
from llama_index.bridge.langchain import ConditionalPromptSelector as LangchainSelector
from llama_index.llms.base import LLM, ChatMessage
from llama_index.llms.generic_utils import messages_to_prompt, prompt_to_messages
from llama_index.llms.langchain_utils import from_lc_messages
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.utils import get_template_vars
from llama_index.types import BaseOutputParser


class BasePromptTemplate(BaseModel, ABC):
    metadata: Dict[str, Any]
    template_vars: List[str]
    kwargs: Dict[str, str]
    output_parser: Optional[BaseOutputParser]

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def partial_format(self, **kwargs: Any) -> "BasePromptTemplate":
        ...

    @abstractmethod
    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        ...

    @abstractmethod
    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        ...


class ChatPromptTemplate(BasePromptTemplate):
    message_templates: List[ChatMessage]

    def __init__(
        self,
        message_templates: List[ChatMessage],
        prompt_type: str = PromptType.CUSTOM,
        output_parser: Optional[BaseOutputParser] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if metadata is None:
            metadata = {}
        metadata["prompt_type"] = prompt_type

        template_vars = []
        for message_template in message_templates:
            template_vars.extend(get_template_vars(message_template.content or ""))

        super().__init__(
            message_templates=message_templates,
            kwargs=kwargs,
            metadata=metadata,
            output_parser=output_parser,
            template_vars=template_vars,
        )

    def partial_format(self, **kwargs: Any) -> "ChatPromptTemplate":
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)
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
            **self.kwargs,
            **kwargs,
        }

        messages = []
        for message_template in self.message_templates:
            template_vars = get_template_vars(message_template.content or "")
            relevant_kwargs = {
                k: v for k, v in all_kwargs.items() if k in template_vars
            }
            content_template = message_template.content or ""
            content = content_template.format(**relevant_kwargs)

            message = message_template.copy()
            message.content = content
            messages.append(message)

        return messages


class PromptTemplate(BasePromptTemplate):
    template: str

    def __init__(
        self,
        template: str,
        prompt_type: str = PromptType.CUSTOM,
        output_parser: Optional[BaseOutputParser] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if metadata is None:
            metadata = {}
        metadata["prompt_type"] = prompt_type

        template_vars = get_template_vars(template)

        super().__init__(
            template=template,
            template_vars=template_vars,
            kwargs=kwargs,
            metadata=metadata,
            output_parser=output_parser,
        )

    def partial_format(self, **kwargs: Any) -> "PromptTemplate":
        """Partially format the prompt."""
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)
        return prompt

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        del llm  # unused
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }
        return self.template.format(**all_kwargs)

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        del llm  # unused
        prompt = self.format(**kwargs)
        return prompt_to_messages(prompt)


class SelectorPromptTemplate(BasePromptTemplate):
    default_prompt: BasePromptTemplate
    conditionals: Optional[List[Tuple[Callable[[LLM], bool], BasePromptTemplate]]] = (
        None,
    )

    def __init__(
        self,
        default_prompt: BasePromptTemplate,
        conditionals: Optional[
            List[Tuple[Callable[[LLM], bool], BasePromptTemplate]]
        ] = None,
    ):
        metadata = default_prompt.metadata
        kwargs = default_prompt.kwargs
        template_vars = default_prompt.template_vars
        output_parser = default_prompt.output_parser
        super().__init__(
            default_prompt=default_prompt,
            conditionals=conditionals,
            metadata=metadata,
            kwargs=kwargs,
            template_vars=template_vars,
            output_parser=output_parser,
        )

    def _select(self, llm: Optional[LLM] = None) -> BasePromptTemplate:
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
        return prompt.format_messages(**kwargs)


class LangchainPromptTemplate(BasePromptTemplate):
    selector: LangchainSelector

    def __init__(
        self,
        template: Optional[LangchainTemplate] = None,
        selector: Optional[LangchainSelector] = None,
        output_parser: Optional[BaseOutputParser] = None,
        prompt_type: str = PromptType.CUSTOM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if selector is None:
            if template is None:
                raise ValueError("Must provide either template or selector.")
            selector = LangchainSelector(default_prompt=template)
        else:
            if template is not None:
                raise ValueError("Must provide either template or selector.")
            selector = selector

        kwargs = selector.default_prompt.partial_variables
        template_vars = selector.default_prompt.input_variables

        if metadata is None:
            metadata = {}
        metadata["prompt_type"] = prompt_type

        super().__init__(
            selector=selector,
            metadata=metadata,
            kwargs=kwargs,
            template_vars=template_vars,
            output_parser=output_parser,
        )

    def partial_format(self, **kwargs: Any) -> "PromptTemplate":
        """Partially format the prompt."""
        default_prompt = self.selector.default_prompt.partial(**kwargs)
        conditionals = [
            (condition, prompt.partial(**kwargs))
            for condition, prompt in self.selector.conditionals
        ]
        lc_selector = LangchainSelector(
            default_prompt=default_prompt, conditionals=conditionals
        )
        return LangchainPromptTemplate(selector=lc_selector)

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        template = self.selector.get_prompt(llm=llm)
        return template.format(**kwargs)

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        template = self.selector.get_prompt(llm=llm)
        lc_prompt_value = template.format_prompt(**kwargs)
        lc_messages = lc_prompt_value.to_messages()
        messages = from_lc_messages(lc_messages)
        return messages


# NOTE: only for backwards compatibility
Prompt = PromptTemplate
