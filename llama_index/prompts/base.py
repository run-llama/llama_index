"""Prompts."""


from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from llama_index.bridge.pydantic import Field

if TYPE_CHECKING:
    from llama_index.bridge.langchain import BasePromptTemplate as LangchainTemplate
    from llama_index.bridge.langchain import (
        ConditionalPromptSelector as LangchainSelector,
    )
from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.base import LLM, ChatMessage
from llama_index.llms.generic_utils import messages_to_prompt, prompt_to_messages
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.utils import get_template_vars
from llama_index.types import BaseOutputParser


class BasePromptTemplate(BaseModel, ABC):
    metadata: Dict[str, Any]
    template_vars: List[str]
    kwargs: Dict[str, str]
    output_parser: Optional[BaseOutputParser]
    template_var_mappings: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Template variable mappings (Optional)."
    )
    function_mappings: Optional[Dict[str, Callable]] = Field(
        default_factory=dict,
        description=(
            "Function mappings (Optional). This is a mapping from template "
            "variable names to functions that take in the current kwargs and "
            "return a string."
        ),
    )

    def _map_template_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """For keys in template_var_mappings, swap in the right keys."""
        template_var_mappings = self.template_var_mappings or {}
        return {template_var_mappings.get(k, k): v for k, v in kwargs.items()}

    def _map_function_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """For keys in function_mappings, compute values and combine w/ kwargs.

        Users can pass in functions instead of fixed values as format variables.
        For each function, we call the function with the current kwargs,
        get back the value, and then use that value in the template
        for the corresponding format variable.

        """
        function_mappings = self.function_mappings or {}
        # first generate the values for the functions
        new_kwargs = {}
        for k, v in function_mappings.items():
            # TODO: figure out what variables to pass into each function
            # is it the kwargs specified during query time? just the fixed kwargs?
            # all kwargs?
            new_kwargs[k] = v(**kwargs)

        # then, add the fixed variables only if not in new_kwargs already
        # (implying that function mapping will override fixed variables)
        for k, v in kwargs.items():
            if k not in new_kwargs:
                new_kwargs[k] = v

        return new_kwargs

    def _map_all_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map both template and function variables.

        We (1) first call function mappings to compute functions,
        and then (2) call the template_var_mappings.

        """
        # map function
        new_kwargs = self._map_function_vars(kwargs)
        # map template vars (to point to existing format vars in string template)
        return self._map_template_vars(new_kwargs)

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

    @abstractmethod
    def get_template(self, llm: Optional[LLM] = None) -> str:
        ...


class PromptTemplate(BasePromptTemplate):
    template: str

    def __init__(
        self,
        template: str,
        prompt_type: str = PromptType.CUSTOM,
        output_parser: Optional[BaseOutputParser] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
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
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )

    def partial_format(self, **kwargs: Any) -> "PromptTemplate":
        """Partially format the prompt."""
        # NOTE: this is a hack to get around deepcopy failing on output parser
        output_parser = self.output_parser
        self.output_parser = None

        # get function and fixed kwargs, and add that to a copy
        # of the current prompt object
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)

        # NOTE: put the output parser back
        prompt.output_parser = output_parser
        return prompt

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        del llm  # unused
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }

        mapped_all_kwargs = self._map_all_vars(all_kwargs)
        prompt = self.template.format(**mapped_all_kwargs)
        if self.output_parser is not None:
            prompt = self.output_parser.format(prompt)
        return prompt

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        del llm  # unused
        prompt = self.format(**kwargs)
        return prompt_to_messages(prompt)

    def get_template(self, llm: Optional[LLM] = None) -> str:
        return self.template


class ChatPromptTemplate(BasePromptTemplate):
    message_templates: List[ChatMessage]

    def __init__(
        self,
        message_templates: List[ChatMessage],
        prompt_type: str = PromptType.CUSTOM,
        output_parser: Optional[BaseOutputParser] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
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
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )

    def partial_format(self, **kwargs: Any) -> "ChatPromptTemplate":
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)
        return prompt

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        del llm  # unused
        messages = self.format_messages(**kwargs)
        return messages_to_prompt(messages)

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        del llm  # unused
        """Format the prompt into a list of chat messages."""
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }
        mapped_all_kwargs = self._map_all_vars(all_kwargs)

        messages: List[ChatMessage] = []
        for message_template in self.message_templates:
            template_vars = get_template_vars(message_template.content or "")
            relevant_kwargs = {
                k: v for k, v in mapped_all_kwargs.items() if k in template_vars
            }
            content_template = message_template.content or ""

            # if there's mappings specified, make sure those are used
            content = content_template.format(**relevant_kwargs)

            message: ChatMessage = message_template.copy()
            message.content = content
            messages.append(message)

        if self.output_parser is not None:
            messages = self.output_parser.format_messages(messages)

        return messages

    def get_template(self, llm: Optional[LLM] = None) -> str:
        return messages_to_prompt(self.message_templates)


class SelectorPromptTemplate(BasePromptTemplate):
    default_template: BasePromptTemplate
    conditionals: Optional[
        List[Tuple[Callable[[LLM], bool], BasePromptTemplate]]
    ] = None

    def __init__(
        self,
        default_template: BasePromptTemplate,
        conditionals: Optional[
            List[Tuple[Callable[[LLM], bool], BasePromptTemplate]]
        ] = None,
    ):
        metadata = default_template.metadata
        kwargs = default_template.kwargs
        template_vars = default_template.template_vars
        output_parser = default_template.output_parser
        super().__init__(
            default_template=default_template,
            conditionals=conditionals,
            metadata=metadata,
            kwargs=kwargs,
            template_vars=template_vars,
            output_parser=output_parser,
        )

    def select(self, llm: Optional[LLM] = None) -> BasePromptTemplate:
        # ensure output parser is up to date
        self.default_template.output_parser = self.output_parser

        if llm is None:
            return self.default_template

        if self.conditionals is not None:
            for condition, prompt in self.conditionals:
                if condition(llm):
                    # ensure output parser is up to date
                    prompt.output_parser = self.output_parser
                    return prompt

        return self.default_template

    def partial_format(self, **kwargs: Any) -> "SelectorPromptTemplate":
        default_template = self.default_template.partial_format(**kwargs)
        if self.conditionals is None:
            conditionals = None
        else:
            conditionals = [
                (condition, prompt.partial_format(**kwargs))
                for condition, prompt in self.conditionals
            ]
        return SelectorPromptTemplate(
            default_template=default_template, conditionals=conditionals
        )

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        prompt = self.select(llm=llm)
        return prompt.format(**kwargs)

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        prompt = self.select(llm=llm)
        return prompt.format_messages(**kwargs)

    def get_template(self, llm: Optional[LLM] = None) -> str:
        prompt = self.select(llm=llm)
        return prompt.get_template(llm=llm)


class LangchainPromptTemplate(BasePromptTemplate):
    selector: Any
    requires_langchain_llm: bool = False

    def __init__(
        self,
        template: Optional["LangchainTemplate"] = None,
        selector: Optional["LangchainSelector"] = None,
        output_parser: Optional[BaseOutputParser] = None,
        prompt_type: str = PromptType.CUSTOM,
        metadata: Optional[Dict[str, Any]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
        requires_langchain_llm: bool = False,
    ) -> None:
        try:
            from llama_index.bridge.langchain import (
                ConditionalPromptSelector as LangchainSelector,
            )
        except ImportError:
            raise ImportError(
                "Must install `llama_index[langchain]` to use LangchainPromptTemplate."
            )
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
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
            requires_langchain_llm=requires_langchain_llm,
        )

    def partial_format(self, **kwargs: Any) -> "BasePromptTemplate":
        """Partially format the prompt."""
        from llama_index.bridge.langchain import (
            ConditionalPromptSelector as LangchainSelector,
        )

        mapped_kwargs = self._map_all_vars(kwargs)
        default_prompt = self.selector.default_prompt.partial(**mapped_kwargs)
        conditionals = [
            (condition, prompt.partial(**mapped_kwargs))
            for condition, prompt in self.selector.conditionals
        ]
        lc_selector = LangchainSelector(
            default_prompt=default_prompt, conditionals=conditionals
        )

        # copy full prompt object, replace selector
        lc_prompt = deepcopy(self)
        lc_prompt.selector = lc_selector
        return lc_prompt

    def format(self, llm: Optional[LLM] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        from llama_index.llms.langchain import LangChainLLM

        if llm is not None:
            # if llamaindex LLM is provided, and we require a langchain LLM,
            # then error. but otherwise if `requires_langchain_llm` is False,
            # then we can just use the default prompt
            if not isinstance(llm, LangChainLLM) and self.requires_langchain_llm:
                raise ValueError("Must provide a LangChainLLM.")
            elif not isinstance(llm, LangChainLLM):
                lc_template = self.selector.default_prompt
            else:
                lc_template = self.selector.get_prompt(llm=llm.llm)
        else:
            lc_template = self.selector.default_prompt

        # if there's mappings specified, make sure those are used
        mapped_kwargs = self._map_all_vars(kwargs)
        return lc_template.format(**mapped_kwargs)

    def format_messages(
        self, llm: Optional[LLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        from llama_index.llms.langchain import LangChainLLM
        from llama_index.llms.langchain_utils import from_lc_messages

        if llm is not None:
            # if llamaindex LLM is provided, and we require a langchain LLM,
            # then error. but otherwise if `requires_langchain_llm` is False,
            # then we can just use the default prompt
            if not isinstance(llm, LangChainLLM) and self.requires_langchain_llm:
                raise ValueError("Must provide a LangChainLLM.")
            elif not isinstance(llm, LangChainLLM):
                lc_template = self.selector.default_prompt
            else:
                lc_template = self.selector.get_prompt(llm=llm.llm)
        else:
            lc_template = self.selector.default_prompt

        # if there's mappings specified, make sure those are used
        mapped_kwargs = self._map_all_vars(kwargs)
        lc_prompt_value = lc_template.format_prompt(**mapped_kwargs)
        lc_messages = lc_prompt_value.to_messages()
        return from_lc_messages(lc_messages)

    def get_template(self, llm: Optional[LLM] = None) -> str:
        from llama_index.llms.langchain import LangChainLLM

        if llm is not None:
            # if llamaindex LLM is provided, and we require a langchain LLM,
            # then error. but otherwise if `requires_langchain_llm` is False,
            # then we can just use the default prompt
            if not isinstance(llm, LangChainLLM) and self.requires_langchain_llm:
                raise ValueError("Must provide a LangChainLLM.")
            elif not isinstance(llm, LangChainLLM):
                lc_template = self.selector.default_prompt
            else:
                lc_template = self.selector.get_prompt(llm=llm.llm)
        else:
            lc_template = self.selector.default_prompt

        try:
            return str(lc_template.template)  # type: ignore
        except AttributeError:
            return str(lc_template)


# NOTE: only for backwards compatibility
Prompt = PromptTemplate
