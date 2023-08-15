"""Outlines program."""

from typing import Any, Dict, Optional, Type, Union, Generator, TYPE_CHECKING

from pydantic import BaseModel

from abc import abstractmethod
from llama_index.llms.base import LLM, ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_utils import to_openai_function
from llama_index.program.llm_prompt_program import BaseLLMFunctionProgram
from llama_index.prompts.base import Prompt
from llama_index.types import Model
from llama_index.program.utils import create_list_model
from typing import Tuple, Callable


def get_monkey_patch_schema_fn(
    model: Type[BaseModel],
) -> None:
    """Monkey patch json schema."""

    def schema_fn() -> Dict[str, Any]:
        import json

        schema_json = model.schema_json()
        # assume $defs needs to exist, swap from definitions
        schema_json = schema_json.replace("definitions", "$defs")
        return json.loads(schema_json)
        # if "$defs" not in schema_dict:
        #     schema_dict["$defs"] = schema_dict["definitions"]
        # return schema_dict

    return schema_fn


class OutlinesProgram(BaseLLMFunctionProgram[Callable]):
    """Outlines program.

    Args:
        output_cls (Type[Model]): Output class.
        prompt_template_str (str): Prompt template string.
        model (Optional[Callable]): Model (see `outlines` repo for full models to use)
            uses `outlines.models.OpenAICompletion` by default.

    """

    def __init__(
        self,
        output_cls: Type[Model],
        prompt_template_str: str,
        llm: Optional[Callable] = None,
    ) -> None:
        """Outlines program."""
        self._output_cls = output_cls
        if not hasattr(output_cls, "model_json_schema"):
            assert hasattr(output_cls, "schema")
            # monkey patch for older versions of pydantic
            output_cls.model_json_schema = get_monkey_patch_schema_fn(output_cls)

        self._prompt_template_str = prompt_template_str
        import outlines.models as models

        if llm is None:
            llm = models.OpenAICompletion(
                "gpt-3.5-turbo", max_tokens=256, temperature=0.1
            )
        self._llm = llm

    @classmethod
    def from_defaults(
        cls,
        output_cls: Type[Model],
        prompt_template_str: str,
        llm: Optional[Callable] = None,
        **kwargs: Any,
    ) -> "BaseLLMFunctionProgram":
        """Initialize program from defaults."""
        return cls(output_cls, prompt_template_str, llm=llm, **kwargs)

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_cls

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        """Call outlines program."""
        import outlines.text.generate as generate

        formatted_prompt = self._prompt_template_str.format(**kwargs)
        print(self._output_cls)
        sequence = generate.json(
            self._llm,
            self._output_cls,
        )(formatted_prompt)
        parsed = self._output_cls.model_validate_json(sequence)
        return parsed
