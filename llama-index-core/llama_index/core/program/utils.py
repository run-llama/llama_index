"""Program utils."""
import logging
from typing import Any, List, Type, Sequence, Union, Optional, Dict

from llama_index.core.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
)
from llama_index.core.llms.llm import LLM, ToolSelection
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.types import BasePydanticProgram, PydanticProgramMode
from llama_index.core.base.llms.types import ChatResponse

_logger = logging.getLogger(__name__)


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


def create_flexible_model(model: Type[BaseModel]) -> Type[FlexibleModel]:
    """Create a flexible version of the model that allows any fields."""
    return create_model(
        f"Flexible{model.__name__}",
        __base__=FlexibleModel,
        **{field: (Optional[Any], None) for field in model.model_fields},
    )  # type: ignore


def create_list_model(base_cls: Type[BaseModel]) -> Type[BaseModel]:
    """Create a list version of an existing Pydantic object."""
    # NOTE: this is directly taken from
    # https://github.com/jxnl/openai_function_call/blob/main/examples/streaming_multitask/streaming_multitask.py
    # all credits go to the openai_function_call repo

    name = f"{base_cls.__name__}List"
    list_items = (
        List[base_cls],  # type: ignore
        Field(
            default_factory=list,  # type: ignore
            repr=False,
            description=f"List of {base_cls.__name__} items",
        ),
    )

    new_cls = create_model(name, items=list_items)
    new_cls.__doc__ = f"A list of {base_cls.__name__} objects. "

    return new_cls


def get_program_for_llm(
    output_cls: Type[BaseModel],
    prompt: BasePromptTemplate,
    llm: LLM,
    pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    **kwargs: Any,
) -> BasePydanticProgram:
    """Get a program based on the compatible LLM."""
    if pydantic_program_mode == PydanticProgramMode.DEFAULT:
        if llm.metadata.is_function_calling_model:
            from llama_index.core.program.function_program import FunctionCallingProgram

            return FunctionCallingProgram.from_defaults(
                output_cls=output_cls,
                llm=llm,
                prompt=prompt,
                **kwargs,
            )
        else:
            from llama_index.core.program.llm_program import (
                LLMTextCompletionProgram,
            )

            return LLMTextCompletionProgram.from_defaults(
                output_parser=PydanticOutputParser(output_cls=output_cls),
                llm=llm,
                prompt=prompt,
                **kwargs,
            )
    elif pydantic_program_mode == PydanticProgramMode.OPENAI:
        from llama_index.program.openai import (
            OpenAIPydanticProgram,
        )  # pants: no-infer-dep

        return OpenAIPydanticProgram.from_defaults(
            output_cls=output_cls,
            llm=llm,
            prompt=prompt,  # type: ignore
            **kwargs,
        )
    elif pydantic_program_mode == PydanticProgramMode.FUNCTION:
        from llama_index.core.program.function_program import FunctionCallingProgram

        return FunctionCallingProgram.from_defaults(
            output_cls=output_cls,
            llm=llm,
            prompt=prompt,
            **kwargs,
        )

    elif pydantic_program_mode == PydanticProgramMode.LLM:
        from llama_index.core.program.llm_program import LLMTextCompletionProgram

        return LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(output_cls=output_cls),
            llm=llm,
            prompt=prompt,
            **kwargs,
        )
    elif pydantic_program_mode == PydanticProgramMode.LM_FORMAT_ENFORCER:
        try:
            from llama_index.program.lmformatenforcer import (
                LMFormatEnforcerPydanticProgram,
            )  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "This mode requires the `llama-index-program-lmformatenforcer package. Please"
                " install it by running `pip install llama-index-program-lmformatenforcer`."
            )

        return LMFormatEnforcerPydanticProgram.from_defaults(
            output_cls=output_cls,
            llm=llm,
            prompt=prompt,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported pydantic program mode: {pydantic_program_mode}")


def _repair_incomplete_json(json_str: str) -> str:
    """Attempt to repair incomplete JSON strings.

    Args:
        json_str (str): Potentially incomplete JSON string

    Returns:
        str: Repaired JSON string
    """
    if not json_str.strip():
        return "{}"

    # Add missing quotes
    quote_count = json_str.count('"')
    if quote_count % 2 == 1:
        json_str += '"'

    # Add missing braces
    brace_count = json_str.count("{") - json_str.count("}")
    if brace_count > 0:
        json_str += "}" * brace_count

    return json_str


def process_streaming_objects(
    chat_response: ChatResponse,
    output_cls: Type[BaseModel],
    cur_objects: Optional[Sequence[BaseModel]] = None,
    allow_parallel_tool_calls: bool = False,
    flexible_mode: bool = True,
    llm: Optional[FunctionCallingLLM] = None,
) -> Union[BaseModel, List[BaseModel]]:
    """Process streaming response into structured objects.

    Args:
        chat_response (ChatResponse): The chat response to process
        output_cls (Type[BaseModel]): The target output class
        cur_objects (Optional[List[BaseModel]]): Current accumulated objects
        allow_parallel_tool_calls (bool): Whether to allow multiple tool calls
        flexible_mode (bool): Whether to use flexible schema during parsing

    Returns:
        Union[BaseModel, List[BaseModel]]: Processed object(s)
    """
    if flexible_mode:
        # Create flexible version of model that allows partial responses
        partial_output_cls = create_flexible_model(output_cls)
    else:
        partial_output_cls = output_cls  # type: ignore

    # Get tool calls from response, if there are any
    if not chat_response.message.additional_kwargs.get("tool_calls"):
        output_cls_args = [chat_response.message.content]
    else:
        tool_calls: List[ToolSelection] = []
        if not llm:
            raise ValueError("LLM is required to get tool calls")

        if isinstance(chat_response.message.additional_kwargs.get("tool_calls"), list):
            tool_calls = llm.get_tool_calls_from_response(
                chat_response, error_on_no_tool_call=False
            )

        if len(tool_calls) == 0:
            # If no tool calls, return single blank output class
            return partial_output_cls()

        # Extract arguments from tool calls
        output_cls_args = [call.tool_kwargs for call in tool_calls]  # type: ignore

    # Try to parse objects, handling potential incomplete JSON
    objects = []
    for output_cls_arg in output_cls_args:
        try:
            # First try direct validation
            obj = partial_output_cls.model_validate(output_cls_arg)
            objects.append(obj)
        except (ValidationError, ValueError):
            try:
                # Try repairing the JSON if it's a string
                if isinstance(output_cls_arg, str):
                    repaired_json = _repair_incomplete_json(output_cls_arg)
                    obj = partial_output_cls.model_validate_json(repaired_json)
                    objects.append(obj)
                else:
                    raise
            except (ValidationError, ValueError) as e2:
                _logger.debug(f"Validation error during streaming: {e2}")
                # If we have previous objects, keep using those
                if cur_objects:
                    objects = cur_objects  # type: ignore
                else:
                    # Return a blank object if we can't parse anything
                    return partial_output_cls()

    # Update if we have more valid fields than before
    if cur_objects is None or num_valid_fields(objects) >= num_valid_fields(
        cur_objects
    ):
        cur_objects = objects  # type: ignore

    # Try to convert flexible objects to target schema
    new_cur_objects = []
    cur_objects = cur_objects or []
    for o in cur_objects:
        try:
            new_obj = output_cls.model_validate(o.model_dump(exclude_unset=True))
        except ValidationError:
            new_obj = o
        new_cur_objects.append(new_obj)

    if allow_parallel_tool_calls:
        return new_cur_objects
    else:
        if len(new_cur_objects) > 1:
            _logger.warning(
                "Multiple outputs found, returning first one. "
                "If you want to return all outputs, set allow_parallel_tool_calls=True."
            )
        return new_cur_objects[0]


def num_valid_fields(
    obj: Union[BaseModel, Sequence[BaseModel], Dict[str, BaseModel]]
) -> int:
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
