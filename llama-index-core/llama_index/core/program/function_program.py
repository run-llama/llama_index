"""Pydantic program through function calling."""

import logging
from typing import (
    Any,
    Dict,
    Optional,
    Type,
    cast,
    Union,
    List,
    Generator,
    AsyncGenerator,
)

from llama_index.core.bridge.pydantic import (
    ValidationError,
)
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.types import BasePydanticProgram, Model
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.program.utils import (
    FlexibleModel,
    process_streaming_objects,
    num_valid_fields,
)

_logger = logging.getLogger(__name__)


def get_function_tool(output_cls: Type[Model]) -> FunctionTool:
    """Get function tool."""
    schema = output_cls.model_json_schema()
    schema_description = schema.get("description", None)

    # NOTE: this does not specify the schema in the function signature,
    # so instead we'll directly provide it in the fn_schema in the ToolMetadata
    def model_fn(**kwargs: Any) -> Model:
        """Model function."""
        return output_cls(**kwargs)

    return FunctionTool.from_defaults(
        fn=model_fn,
        name=schema["title"],
        description=schema_description,
        fn_schema=output_cls,
    )


class FunctionCallingProgram(BasePydanticProgram[Model]):
    """Function Calling Program.

    Uses function calling LLMs to obtain a structured output.
    """

    def __init__(
        self,
        output_cls: Type[Model],
        llm: FunctionCallingLLM,
        prompt: BasePromptTemplate,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
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
        prompt: Optional[BasePromptTemplate] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "FunctionCallingProgram":
        llm = llm or Settings.llm  # type: ignore
        assert llm is not None

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
            llm=llm,  # type: ignore
            prompt=cast(PromptTemplate, prompt),
            tool_choice=tool_choice,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            verbose=verbose,
        )

    @property
    def output_cls(self) -> Type[Model]:
        return self._output_cls

    @property
    def prompt(self) -> BasePromptTemplate:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        self._prompt = prompt

    def __call__(
        self,
        *args: Any,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Model, List[Model]]:
        llm_kwargs = llm_kwargs or {}
        tool = get_function_tool(self._output_cls)

        messages = self._prompt.format_messages(llm=self._llm, **kwargs)
        messages = self._llm._extend_messages(messages)

        agent_response = self._llm.predict_and_call(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return self._parse_tool_outputs(
            agent_response,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )

    async def acall(
        self,
        *args: Any,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Model, List[Model]]:
        llm_kwargs = llm_kwargs or {}
        tool = get_function_tool(self._output_cls)

        agent_response = await self._llm.apredict_and_call(
            [tool],
            chat_history=self._prompt.format_messages(llm=self._llm, **kwargs),
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return self._parse_tool_outputs(
            agent_response,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )

    def _parse_tool_outputs(
        self,
        agent_response: AgentChatResponse,
        allow_parallel_tool_calls: bool = False,
    ) -> Union[Model, List[Model]]:
        """Parse tool outputs."""
        outputs = [cast(Model, s.raw_output) for s in agent_response.sources]
        if allow_parallel_tool_calls:
            return outputs
        else:
            if len(outputs) > 1:
                _logger.warning(
                    "Multiple outputs found, returning first one. "
                    "If you want to return all outputs, set output_multiple=True."
                )

            return outputs[0]

    def _process_objects(
        self,
        chat_response: ChatResponse,
        output_cls: Type[Model],
        cur_objects: Optional[List[Model]] = None,
    ) -> Union[Model, List[Model]]:
        """Process stream."""
        tool_calls = self._llm.get_tool_calls_from_response(
            chat_response,
            # error_on_no_tool_call=True
            error_on_no_tool_call=False,
        )
        # TODO: change
        if len(tool_calls) == 0:
            # if no tool calls, return single blank output_class
            return output_cls()

        tool_fn_args = [call.tool_kwargs for call in tool_calls]
        objects = [
            output_cls.model_validate(tool_fn_arg) for tool_fn_arg in tool_fn_args
        ]

        if cur_objects is None or num_valid_fields(objects) > num_valid_fields(
            cur_objects
        ):
            cur_objects = objects

        # right now the objects are typed according to a flexible schema
        # try to do a pass to convert the objects to the output_cls
        new_cur_objects = []
        for obj in cur_objects:
            try:
                new_obj = self._output_cls.model_validate(obj.model_dump())
            except ValidationError as e:
                _logger.warning(f"Failed to parse object: {e}")
                new_obj = obj
            new_cur_objects.append(new_obj)

        if self._allow_parallel_tool_calls:
            return new_cur_objects
        else:
            if len(new_cur_objects) > 1:
                _logger.warning(
                    "Multiple outputs found, returning first one. "
                    "If you want to return all outputs, set output_multiple=True."
                )
            return new_cur_objects[0]

    def stream_call(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Generator[
        Union[Model, List[Model], FlexibleModel, List[FlexibleModel]], None, None
    ]:
        """Stream object.

        Returns a generator returning partials of the same object
        or a list of objects until it returns.
        """
        # TODO: we can extend this to non-function calling LLMs as well, coming soon
        if not isinstance(self._llm, FunctionCallingLLM):
            raise ValueError("stream_call is only supported for LLMs.")

        llm_kwargs = llm_kwargs or {}
        tool = get_function_tool(self._output_cls)

        messages = self._prompt.format_messages(llm=self._llm, **kwargs)
        messages = self._llm._extend_messages(messages)

        chat_response_gen = self._llm.stream_chat_with_tools(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )

        cur_objects = None
        for partial_resp in chat_response_gen:
            try:
                objects = process_streaming_objects(
                    partial_resp,
                    self._output_cls,
                    cur_objects=cur_objects,
                    allow_parallel_tool_calls=self._allow_parallel_tool_calls,
                    flexible_mode=True,
                    llm=self._llm,
                )
                cur_objects = objects if isinstance(objects, list) else [objects]
                yield objects
            except Exception as e:
                _logger.warning(f"Failed to parse streaming response: {e}")
                continue

    async def astream_call(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> AsyncGenerator[
        Union[Model, List[Model], FlexibleModel, List[FlexibleModel]], None
    ]:
        """Stream objects.

        Returns a generator returning partials of the same object
        or a list of objects until it returns.
        """
        if not isinstance(self._llm, FunctionCallingLLM):
            raise ValueError("stream_call is only supported for LLMs.")

        tool = get_function_tool(self._output_cls)

        messages = self._prompt.format_messages(llm=self._llm, **kwargs)
        messages = self._llm._extend_messages(messages)

        chat_response_gen = await self._llm.astream_chat_with_tools(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **(llm_kwargs or {}),
        )

        async def gen() -> AsyncGenerator[
            Union[Model, List[Model], FlexibleModel, List[FlexibleModel]], None
        ]:
            cur_objects = None
            async for partial_resp in chat_response_gen:
                try:
                    objects = process_streaming_objects(
                        partial_resp,
                        self._output_cls,
                        cur_objects=cur_objects,
                        allow_parallel_tool_calls=self._allow_parallel_tool_calls,
                        flexible_mode=True,
                        llm=self._llm,
                    )
                    cur_objects = objects if isinstance(objects, list) else [objects]
                    yield objects
                except Exception as e:
                    _logger.warning(f"Failed to parse streaming response: {e}")
                    continue

        return gen()
