"""Pydantic program through function calling."""

import logging
from typing import Any, Dict, Optional, Type, cast, Union, List

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.types import BasePydanticProgram, Model
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.chat_engine.types import AgentChatResponse

_logger = logging.getLogger(__name__)


def _parse_tool_outputs(
    agent_response: AgentChatResponse,
    allow_parallel_tool_calls: bool = False,
) -> List[BaseModel]:
    """Parse tool outputs."""
    outputs = [cast(BaseModel, s.raw_output) for s in agent_response.sources]
    if allow_parallel_tool_calls:
        return outputs
    else:
        if len(outputs) > 1:
            _logger.warning(
                "Multiple outputs found, returning first one. "
                "If you want to return all outputs, set output_multiple=True."
            )

        return outputs[0]


def _get_function_tool(output_cls: Type[Model]) -> FunctionTool:
    """Get function tool."""
    schema = output_cls.schema()
    schema_description = schema.get("description", None)

    # NOTE: this does not specify the schema in the function signature,
    # so instead we'll directly provide it in the fn_schema in the ToolMetadata
    def model_fn(**kwargs: Any) -> BaseModel:
        """Model function."""
        return output_cls(**kwargs)

    return FunctionTool.from_defaults(
        fn=model_fn,
        name=schema["title"],
        description=schema_description,
        fn_schema=output_cls,
    )


class FunctionCallingProgram(BasePydanticProgram[BaseModel]):
    """
    Function Calling Program.

    Uses function calling LLMs to obtain a structured output.

    """

    def __init__(
        self,
        output_cls: Type[Model],
        llm: LLM,
        prompt: BasePromptTemplate,
        tool_choice: Union[str, Dict[str, Any]],
        allow_parallel_tool_calls: bool = False,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        self._output_cls = output_cls
        self._llm = llm
        self._prompt = prompt
        self._verbose = verbose
        self._allow_parallel_tool_calls = allow_parallel_tool_calls
        self._tool_choice = tool_choice

    @classmethod
    def from_defaults(
        cls,
        output_cls: Type[Model],
        prompt_template_str: Optional[str] = None,
        prompt: Optional[PromptTemplate] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "FunctionCallingProgram":
        llm = llm or Settings.llm
        if not llm.metadata.is_function_calling_model:
            raise ValueError(
                f"Model name {llm.metadata.model_name} does not support "
                "function calling API. "
            )

        if prompt is None and prompt_template_str is None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None and prompt_template_str is not None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt_template_str is not None:
            prompt = PromptTemplate(prompt_template_str)

        return cls(
            output_cls=output_cls,
            llm=llm,
            prompt=cast(PromptTemplate, prompt),
            tool_choice=tool_choice,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            verbose=verbose,
        )

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_cls

    @property
    def prompt(self) -> BasePromptTemplate:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        self._prompt = prompt

    def __call__(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Union[Model, List[Model]]:
        llm_kwargs = llm_kwargs or {}
        tool = _get_function_tool(self._output_cls)

        messages = self._prompt.format_messages(llm=self._llm, **kwargs)
        messages = self._llm._extend_messages(messages)

        agent_response = self._llm.predict_and_call(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return _parse_tool_outputs(
            agent_response,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )

    async def acall(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Union[Model, List[Model]]:
        llm_kwargs = llm_kwargs or {}
        tool = _get_function_tool(self._output_cls)

        agent_response = await self._llm.apredict_and_call(
            [tool],
            chat_history=self._prompt.format_messages(llm=self._llm, **kwargs),
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return _parse_tool_outputs(
            agent_response,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )
