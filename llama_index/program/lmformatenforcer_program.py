import json
from typing import Any, Dict, Optional, Type, Union, cast

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.program.llm_prompt_program import BaseLLMFunctionProgram
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.lmformatenforcer_utils import (
    activate_lm_format_enforcer,
    build_lm_format_enforcer_function,
)


class LMFormatEnforcerPydanticProgram(BaseLLMFunctionProgram):
    """
    A lm-format-enforcer-based function that returns a pydantic model.

    In LMFormatEnforcerPydanticProgram, prompt_template_str can also have a {json_schema} parameter
    that will be automatically filled by the json_schema of output_cls.
    Note: this interface is not yet stable.
    """

    def __init__(
        self,
        output_cls: Type[BaseModel],
        prompt_template_str: str,
        llm: Optional[Union[LlamaCPP, HuggingFaceLLM]] = None,
        verbose: bool = False,
    ):
        try:
            import lmformatenforcer
        except ImportError as e:
            raise ImportError(
                "lm-format-enforcer package not found."
                "please run `pip install lm-format-enforcer`"
            ) from e

        if llm is None:
            try:
                from llama_index.llms import LlamaCPP

                llm = LlamaCPP()
            except ImportError as e:
                raise ImportError(
                    "llama.cpp package not found."
                    "please run `pip install llama-cpp-python`"
                ) from e

        self.llm = llm

        self._prompt_template_str = prompt_template_str
        self._output_cls = output_cls
        self._verbose = verbose
        json_schema_parser = lmformatenforcer.JsonSchemaParser(self.output_cls.schema())
        self._token_enforcer_fn = build_lm_format_enforcer_function(
            self.llm, json_schema_parser
        )

    @classmethod
    def from_defaults(
        cls,
        output_cls: Type[BaseModel],
        prompt_template_str: Optional[str] = None,
        prompt: Optional[PromptTemplate] = None,
        llm: Optional[Union["LlamaCPP", "HuggingFaceLLM"]] = None,
        **kwargs: Any,
    ) -> "BaseLLMFunctionProgram":
        """From defaults."""
        if prompt is None and prompt_template_str is None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None and prompt_template_str is not None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None:
            prompt_template_str = prompt.template
        prompt_template_str = cast(str, prompt_template_str)
        return cls(
            output_cls,
            prompt_template_str,
            llm=llm,
            **kwargs,
        )

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_cls

    def __call__(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        llm_kwargs = llm_kwargs or {}
        # While the format enforcer is active, any calls to the llm will have the format enforced.
        with activate_lm_format_enforcer(self.llm, self._token_enforcer_fn):
            json_schema_str = json.dumps(self.output_cls.schema())
            full_str = self._prompt_template_str.format(
                *args, **kwargs, json_schema=json_schema_str
            )
            output = self.llm.complete(full_str, **llm_kwargs)
            text = output.text
            return self.output_cls.parse_raw(text)
