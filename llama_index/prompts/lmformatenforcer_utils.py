from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llm import LLM

if TYPE_CHECKING:
    from lmformatenforcer import CharacterLevelParser


def build_lm_format_enforcer_function(
    llm: LLM, character_level_parser: "CharacterLevelParser"
) -> Callable:
    """Prepare for using the LM format enforcer.
    This builds the processing function that will be injected into the LLM to
    activate the LM Format Enforcer.
    """
    if isinstance(llm, HuggingFaceLLM):
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        return build_transformers_prefix_allowed_tokens_fn(
            llm._tokenizer, character_level_parser
        )
    if isinstance(llm, LlamaCPP):
        from llama_cpp import LogitsProcessorList
        from lmformatenforcer.integrations.llamacpp import (
            build_llamacpp_logits_processor,
        )

        return LogitsProcessorList(
            [build_llamacpp_logits_processor(llm._model, character_level_parser)]
        )
    raise ValueError("Unsupported LLM type")


@contextmanager
def activate_lm_format_enforcer(
    llm: LLM, lm_format_enforcer_fn: Callable
) -> Iterator[None]:
    """Activate the LM Format Enforcer for the given LLM.

    with activate_lm_format_enforcer(llm, lm_format_enforcer_fn):
        llm.complete(...)
    """
    if isinstance(llm, HuggingFaceLLM):
        generate_kwargs_key = "prefix_allowed_tokens_fn"
    elif isinstance(llm, LlamaCPP):
        generate_kwargs_key = "logits_processor"
    else:
        raise ValueError("Unsupported LLM type")
    llm.generate_kwargs[generate_kwargs_key] = lm_format_enforcer_fn

    try:
        # This is where the user code will run
        yield
    finally:
        # We remove the token enforcer function from the generate_kwargs at the end
        # in case other code paths use the same llm object.
        del llm.generate_kwargs[generate_kwargs_key]
