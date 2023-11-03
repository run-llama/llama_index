"""Program utils."""

from typing import Any, List, Type

from llama_index.bridge.pydantic import BaseModel, Field, create_model
from llama_index.llms.base import LLM
from llama_index.output_parsers.pydantic import PydanticOutputParser
from llama_index.prompts.base import PromptTemplate
from llama_index.types import BasePydanticProgram, PydanticProgramMode


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
    output_cls: BaseModel,
    prompt: PromptTemplate,
    llm: LLM,
    pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    **kwargs: Any,
) -> BasePydanticProgram:
    """Get a program based on the compatible LLM."""
    if pydantic_program_mode == PydanticProgramMode.DEFAULT:
        # in default mode, we try to use the OpenAI program if available else
        # we fall back to the LLM program
        try:
            from llama_index.program.openai_program import OpenAIPydanticProgram

            return OpenAIPydanticProgram.from_defaults(
                output_cls=output_cls,
                llm=llm,
                prompt=prompt,
                **kwargs,
            )
        except ValueError:
            from llama_index.program.llm_program import LLMTextCompletionProgram

            return LLMTextCompletionProgram.from_defaults(
                output_parser=PydanticOutputParser(output_cls=output_cls),
                llm=llm,
                prompt=prompt,
                **kwargs,
            )
    elif pydantic_program_mode == PydanticProgramMode.OPENAI:
        from llama_index.program.openai_program import OpenAIPydanticProgram

        return OpenAIPydanticProgram.from_defaults(
            output_cls=output_cls,
            llm=llm,
            prompt=prompt,
            **kwargs,
        )
    elif pydantic_program_mode == PydanticProgramMode.LLM:
        from llama_index.program.llm_program import LLMTextCompletionProgram

        return LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(output_cls=output_cls),
            llm=llm,
            prompt=prompt,
            **kwargs,
        )
    elif pydantic_program_mode == PydanticProgramMode.LM_FORMAT_ENFORCER:
        from llama_index.program.lmformatenforcer_program import (
            LMFormatEnforcerPydanticProgram,
        )

        return LMFormatEnforcerPydanticProgram.from_defaults(
            output_cls=output_cls,
            llm=llm,
            prompt=prompt,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported pydantic program mode: {pydantic_program_mode}")
