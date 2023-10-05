from typing import List

from llama_index.prompts.base import BasePromptTemplate


def get_empty_prompt_txt(prompt: BasePromptTemplate) -> str:
    """Get empty prompt text.

    Substitute empty strings in parts of the prompt that have
    not yet been filled out. Skip variables that have already
    been partially formatted. This is used to compute the initial tokens.

    """
    partial_kargs = prompt.kwargs
    empty_kwargs = {v: "" for v in prompt.template_vars if v not in partial_kargs}
    all_kwargs = {**partial_kargs, **empty_kwargs}
    return prompt.format(llm=None, **all_kwargs)


def get_biggest_prompt(prompts: List[BasePromptTemplate]) -> BasePromptTemplate:
    """Get biggest prompt.

    Oftentimes we need to fetch the biggest prompt, in order to
    be the most conservative about chunking text. This
    is a helper utility for that.

    """
    empty_prompt_txts = [get_empty_prompt_txt(prompt) for prompt in prompts]
    empty_prompt_txt_lens = [len(txt) for txt in empty_prompt_txts]
    return prompts[empty_prompt_txt_lens.index(max(empty_prompt_txt_lens))]
