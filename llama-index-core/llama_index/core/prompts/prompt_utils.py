from typing import List

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.prompts.base import BasePromptTemplate


def get_empty_prompt_txt(prompt: BasePromptTemplate) -> str:
    """
    Get empty prompt text.

    Substitute empty strings in parts of the prompt that have
    not yet been filled out. Skip variables that have already
    been partially formatted. This is used to compute the initial tokens.

    """
    partial_kargs = prompt.kwargs
    empty_kwargs = {v: "" for v in prompt.template_vars if v not in partial_kargs}
    all_kwargs = {**partial_kargs, **empty_kwargs}
    return prompt.format(llm=None, **all_kwargs)


def get_empty_prompt_messages(prompt: BasePromptTemplate) -> list[ChatMessage]:
    """
    Get empty prompt message.

    Substitute empty strings in parts of the prompt that have
    not yet been filled out. Skip variables that have already
    been partially formatted. This is used to compute the initial tokens.
    """
    partial_kargs = prompt.kwargs
    empty_kwargs = {v: "" for v in prompt.template_vars if v not in partial_kargs}
    all_kwargs = {**partial_kargs, **empty_kwargs}
    return prompt.format_messages(llm=None, **all_kwargs)


def get_biggest_prompt(prompts: List[BasePromptTemplate]) -> BasePromptTemplate:
    """
    Get biggest prompt.

    Oftentimes we need to fetch the biggest prompt, in order to
    be the most conservative about chunking text. This
    is a helper utility for that.

    """
    return max(prompts, key=lambda p: len(get_empty_prompt_txt(p)))


def get_biggest_chat_prompt(prompts: List[BasePromptTemplate]) -> BasePromptTemplate:
    """
    Get biggest chat prompt.

    Oftentimes we need to fetch the biggest prompt, in order to
    be the most conservative about chunking text. This
    is a helper utility for that.

    Critically, this function will count tokens in non text blocks, whereas get_biggest_prompt
    only counts text tokens.
    """
    return max(
        prompts,
        key=lambda p: sum([m.estimate_tokens() for m in get_empty_prompt_messages(p)]),
    )
