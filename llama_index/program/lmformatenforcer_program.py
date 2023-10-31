import json
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, Union, cast

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.program.llm_prompt_program import BaseLLMFunctionProgram
from llama_index.prompts.base import PromptTemplate


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
                "lm-format-enforcer package not found." "please run `pip install lm-format-enforcer`"
            ) from e
        
        if llm is None:
            try:
                from llama_index.llms import LlamaCPP
                llm = LlamaCPP()
            except ImportError as e:
                raise ImportError(
                    "llama.cpp package not found." "please run `pip install llama-cpp-python`"
                ) from e
        
        self.llm = llm
        
        self._prompt_template_str = prompt_template_str
        self._output_cls = output_cls
        self._verbose = verbose
        self._token_enforcer_fn = self._build_token_enforcer_function()

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

    def _build_token_enforcer_function(self) -> Callable:
        import lmformatenforcer
        json_schema_parser = lmformatenforcer.JsonSchemaParser(self.output_cls.schema())
        if isinstance(self.llm, HuggingFaceLLM):
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            return build_transformers_prefix_allowed_tokens_fn(self.llm._tokenizer, json_schema_parser)
        if isinstance(self.llm, LlamaCPP):
            from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor
            from llama_cpp import LogitsProcessorList
            return LogitsProcessorList([build_llamacpp_logits_processor(self.llm._model, json_schema_parser)])
        raise ValueError("Unsupported LLM type")
    
    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_cls

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        
        generate_kwargs_key = ''
        if isinstance(self.llm, HuggingFaceLLM):
            generate_kwargs_key = "prefix_allowed_tokens_fn"
        elif isinstance(self.llm, LlamaCPP):
            generate_kwargs_key = "logits_processor"
        
        try:
            self.llm.generate_kwargs[generate_kwargs_key] = self._token_enforcer_fn
            json_schema_str = json.dumps(self.output_cls.schema())
            full_str = self._prompt_template_str.format(*args, **kwargs, json_schema=json_schema_str)
            output = self.llm.complete(full_str)
            text = output.text
            return self.output_cls.parse_raw(text)
        finally:
            # We remove the token enforcer function from the generate_kwargs at the end
            # in case other code paths use the same llm object.
            del self.llm.generate_kwargs[generate_kwargs_key]
        
