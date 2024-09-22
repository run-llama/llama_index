"""Prompts."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from typing_extensions import Annotated

from llama_index.core.bridge.pydantic import (
    Field,
    WithJsonSchema,
    PlainSerializer,
    SerializeAsAny,
)

if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import (
        BasePromptTemplate as LangchainTemplate,
    )

    # pants: no-infer-dep
    from llama_index.core.bridge.langchain import (
        ConditionalPromptSelector as LangchainSelector,
    )
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.base.query_pipeline.query import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as default_messages_to_prompt,
)
from llama_index.core.base.llms.generic_utils import (
    prompt_to_messages,
)
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.utils import get_template_vars
from llama_index.core.types import BaseOutputParser


AnnotatedCallable = Annotated[
    Callable,
    WithJsonSchema({"type": "string"}),
    WithJsonSchema({"type": "string"}),
    PlainSerializer(lambda x: f"{x.__module__}.{x.__name__}", return_type=str),
]


class BasePromptTemplate(ChainableMixin, BaseModel, ABC):  # type: ignore[no-redef]
    model_config = ConfigDict(arbitrary_types_allowed=True)
    metadata: Dict[str, Any]
    template_vars: List[str]
    kwargs: Dict[str, str]
    output_parser: Optional[BaseOutputParser]
    template_var_mappings: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Template variable mappings (Optional)."
    )
    function_mappings: Optional[Dict[str, AnnotatedCallable]] = Field(
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

    @abstractmethod
    def partial_format(self, **kwargs: Any) -> "BasePromptTemplate":
        ...

    @abstractmethod
    def format(self, llm: Optional[BaseLLM] = None, **kwargs: Any) -> str:
        ...

    @abstractmethod
    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        ...

    @abstractmethod
    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        ...

    def _as_query_component(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> QueryComponent:
        """As query component."""
        return PromptComponent(prompt=self, format_messages=False, llm=llm)


class PromptTemplate(BasePromptTemplate):  # type: ignore[no-redef]
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
        self.output_parser = output_parser
        return prompt

    def format(
        self,
        llm: Optional[BaseLLM] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        **kwargs: Any,
    ) -> str:
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

        if completion_to_prompt is not None:
            prompt = completion_to_prompt(prompt)

        return prompt

    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        del llm  # unused
        prompt = self.format(**kwargs)
        return prompt_to_messages(prompt)

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        return self.template


class ChatPromptTemplate(BasePromptTemplate):  # type: ignore[no-redef]
    message_templates: List[ChatMessage]

    def __init__(
        self,
        message_templates: Sequence[ChatMessage],
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

    @classmethod
    def from_messages(
        cls,
        message_templates: Union[List[Tuple[str, str]], List[ChatMessage]],
        **kwargs: Any,
    ) -> "ChatPromptTemplate":
        """From messages."""
        if isinstance(message_templates[0], tuple):
            message_templates = [
                ChatMessage.from_str(role=role, content=content)  # type: ignore[arg-type]
                for role, content in message_templates
            ]
        return cls(message_templates=message_templates, **kwargs)  # type: ignore[arg-type]

    def partial_format(self, **kwargs: Any) -> "ChatPromptTemplate":
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)
        return prompt

    def format(
        self,
        llm: Optional[BaseLLM] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        **kwargs: Any,
    ) -> str:
        del llm  # unused
        messages = self.format_messages(**kwargs)

        if messages_to_prompt is not None:
            return messages_to_prompt(messages)

        return default_messages_to_prompt(messages)

    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
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
            message_content = message_template.content or ""

            template_vars = get_template_vars(message_content)
            relevant_kwargs = {
                k: v for k, v in mapped_all_kwargs.items() if k in template_vars
            }
            content_template = message_template.content or ""

            # if there's mappings specified, make sure those are used
            content = content_template.format(**relevant_kwargs)

            message: ChatMessage = message_template.model_copy()
            message.content = content
            messages.append(message)

        if self.output_parser is not None:
            messages = self.output_parser.format_messages(messages)

        return messages

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        return default_messages_to_prompt(self.message_templates)

    def _as_query_component(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> QueryComponent:
        """As query component."""
        return PromptComponent(prompt=self, format_messages=True, llm=llm)


class SelectorPromptTemplate(BasePromptTemplate):  # type: ignore[no-redef]
    default_template: SerializeAsAny[BasePromptTemplate]
    conditionals: Optional[
        Sequence[Tuple[Callable[[BaseLLM], bool], BasePromptTemplate]]
    ] = None

    def __init__(
        self,
        default_template: BasePromptTemplate,
        conditionals: Optional[
            Sequence[Tuple[Callable[[BaseLLM], bool], BasePromptTemplate]]
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

    def select(self, llm: Optional[BaseLLM] = None) -> BasePromptTemplate:
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

    def format(self, llm: Optional[BaseLLM] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        prompt = self.select(llm=llm)
        return prompt.format(**kwargs)

    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        prompt = self.select(llm=llm)
        return prompt.format_messages(**kwargs)

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        prompt = self.select(llm=llm)
        return prompt.get_template(llm=llm)


class LangchainPromptTemplate(BasePromptTemplate):  # type: ignore[no-redef]
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
            from llama_index.core.bridge.langchain import (
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
        from llama_index.core.bridge.langchain import (
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

    def format(self, llm: Optional[BaseLLM] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        from llama_index.llms.langchain import LangChainLLM  # pants: no-infer-dep

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
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        from llama_index.llms.langchain import LangChainLLM  # pants: no-infer-dep
        from llama_index.llms.langchain.utils import (
            from_lc_messages,
        )  # pants: no-infer-dep

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

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        from llama_index.llms.langchain import LangChainLLM  # pants: no-infer-dep

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


class PromptComponent(QueryComponent):
    """Prompt component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: SerializeAsAny[BasePromptTemplate] = Field(..., description="Prompt")
    llm: Optional[SerializeAsAny[BaseLLM]] = Field(
        default=None, description="LLM to use for formatting prompt."
    )
    format_messages: bool = Field(
        default=False,
        description="Whether to format the prompt into a list of chat messages.",
    )

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        keys = list(input.keys())
        for k in keys:
            input[k] = validate_and_convert_stringable(input[k])
        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        if self.format_messages:
            output: Union[str, List[ChatMessage]] = self.prompt.format_messages(
                llm=self.llm, **kwargs
            )
        else:
            output = self.prompt.format(llm=self.llm, **kwargs)
        return {"prompt": output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # NOTE: no native async for prompt
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys(
            set(self.prompt.template_vars) - set(self.prompt.kwargs)
        )

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"prompt"})
