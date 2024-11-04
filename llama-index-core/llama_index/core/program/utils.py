"""Program utils."""

from typing import Any, List, Type

from llama_index.core.bridge.pydantic import BaseModel, Field, create_model
from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.types import BasePydanticProgram, PydanticProgramMode


def create_list_model(base_cls: Type[BaseModel]) -> Type[BaseModel]:
    """Create a list version of an existing Pydantic object."""
    # NOTE: this is directly taken from
    # https://github.com/jxnl/openai_function_call/blob/main/examples/streaming_multitask/streaming_multitask.py
    # all credits go to the openai_function_call repo

    name = f"{base_cls.__name__}List"
    list_items = (
        List[base_cls],  # type: ignore
        Field(
            default_factory=list,
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
