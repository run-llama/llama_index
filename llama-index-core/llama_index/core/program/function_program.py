"""Pydantic program through function calling."""

import logging
from typing import Any, Dict, Optional, Type, cast, Union, List, Generator, AsyncGenerator, get_args, get_origin

from llama_index.core.bridge.pydantic import BaseModel, create_model, Field, ValidationError
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import ChatResponseGen
from llama_index.core.llms.function_calling import FunctionCallingLLM
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


# def create_partial_model(model: Type[BaseModel], processed_models=None) -> Type[BaseModel]:
#     """
#     Dynamically create a new Pydantic model with all fields as Optional,
#     including nested models.
#     """
#     if processed_models is None:
#         processed_models = {}
    
#     if model.__name__ in processed_models:
#         return processed_models[model.__name__]
    
#     fields = {}
#     for name, field in model.__fields__.items():
#         field_type = field.outer_type_
        
#         if get_origin(field_type) is Union and type(None) in get_args(field_type):
#             # Already Optional, keep as is
#             new_type = field_type
#         elif get_origin(field_type) is List:
#             item_type = get_args(field_type)[0]
#             if isinstance(item_type, type) and issubclass(item_type, BaseModel):
#                 nested_partial = create_partial_model(item_type, processed_models)
#                 new_type = Optional[List[nested_partial]]
#             else:
#                 new_type = Optional[List[Optional[item_type]]]
#         elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
#             nested_partial = create_partial_model(field_type, processed_models)
#             new_type = Optional[nested_partial]
#         else:
#             new_type = Optional[field_type]
        
#         # Use Field with default=None to make it optional
#         fields[name] = (new_type, Field(default=None))
    
#     partial_model = create_model(f"Partial{model.__name__}", **fields)
#     processed_models[model.__name__] = partial_model
    
#     # Update the model's schema to remove 'required' fields
#     def custom_model_json_schema(*args, **kwargs):
#         schema = partial_model.model_json_schema(*args, **kwargs)
#         schema.pop('required', None)
#         for prop in schema.get('properties', {}).values():
#             if isinstance(prop, dict):
#                 prop.pop('required', None)
#         return schema

#     partial_model.model_json_schema = custom_model_json_schema
    
#     return partial_model

class FlexibleModel(BaseModel):
    class Config:
        extra = "allow"

def create_flexible_model(model: Type[BaseModel]) -> Type[FlexibleModel]:
    """Create a flexible version of the model that allows any fields."""
    return create_model(
        f"Flexible{model.__name__}",
        __base__=FlexibleModel,
        **{field: (Optional[Any], None) for field in model.__fields__}
    )

def num_valid_fields(obj: BaseModel) -> int:
    """
    Recursively count the number of fields in a Pydantic object (including nested objects) that aren't None.

    Args:
        obj (Any): A Pydantic model instance or any other object.

    Returns:
        int: The number of fields that have non-None values.
    """
    if isinstance(obj, BaseModel):
        count = 0
        for value in obj.__dict__.values():
            if isinstance(value, (list, tuple)):
                count += sum(num_valid_fields(item) for item in value)
            elif isinstance(value, dict):
                count += sum(num_valid_fields(item) for item in value.values())
            elif isinstance(value, BaseModel):
                count += num_valid_fields(value)
            elif value is not None:
                count += 1
        return count
    elif isinstance(obj, (list, tuple)):
        return sum(num_valid_fields(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(num_valid_fields(item) for item in obj.values())
    else:
        return 1 if obj is not None else 0


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

    def _process_stream(self, chat_response_gen: ChatResponseGen) -> Generator[Union[Model, List[Model]], None, None]:
        """Process stream."""
        # NOTE: create a new class that treats all its fields as optional
        # inspired by instructor
        # https://python.useinstructor.com/concepts/partial/#understanding-partial-responses
        partial_output_cls = create_flexible_model(self._output_cls)

        print(partial_output_cls.schema())

        cur_objects = None
        for partial_resp in chat_response_gen:
            tool_calls = self._llm.get_tool_calls_from_response(
                partial_resp, error_on_no_tool_call=True
            )
            tool_fn_args = [call.tool_kwargs for call in tool_calls]
            objects = [partial_output_cls.parse_obj(tool_fn_arg) for tool_fn_arg in tool_fn_args]

            if cur_objects is None or num_valid_fields(objects) > num_valid_fields(cur_objects):
                cur_objects = objects

            # right now the objects are typed according to a flexible schema
            # try to do a pass to convert the objects to the output_cls
            new_cur_objects = []
            for obj in cur_objects:
                try:
                    new_obj = self._output_cls.parse_obj(obj.dict())
                except ValidationError as e:
                    _logger.warning(f"Failed to parse object: {e}")
                    new_obj = obj
                new_cur_objects.append(new_obj)

            if self._allow_parallel_tool_calls:
                yield new_cur_objects
            else:
                if len(new_cur_objects) > 1:
                    _logger.warning(
                        "Multiple outputs found, returning first one. "
                        "If you want to return all outputs, set output_multiple=True."
                    )
                yield new_cur_objects[0]

    def stream_call(
        self, 
        *args: Any, 
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Generator[Union[Model, List[Model]], None, None]:
        """Stream objects."""

        # TODO: we can extend this to non-function calling LLMs as well, coming soon
        if not isinstance(self._llm, FunctionCallingLLM):
            raise ValueError("stream_call is only supported for LLMs.")
        
        llm_kwargs = llm_kwargs or {}
        tool = _get_function_tool(self._output_cls)

        messages = self._prompt.format_messages(llm=self._llm, **kwargs)
        messages = self._llm._extend_messages(messages)

        chat_response_gen = self._llm.stream_chat_with_tools(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return self._process_stream(chat_response_gen)

            
        # raise NotImplementedError("stream_call is not supported by default.")

    # async def astream_call(
    #     self, *args: Any, **kwargs: Any
    # ) -> AsyncGenerator[Model, None]:
    #     raise NotImplementedError("astream_call is not supported by default.")