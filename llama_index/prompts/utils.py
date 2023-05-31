from typing import List
from llama_index.prompts.base import Prompt


def get_empty_prompt_txt(prompt: Prompt) -> str:
    """Get empty prompt text.

    Substitute empty strings in parts of the prompt that have
    not yet been filled out. Skip variables that have already
    been partially formatted. This is used to compute the initial tokens.

    """
    fmt_dict = {
        v: ""
        for v in prompt.get_langchain_prompt().input_variables
        if v not in prompt.partial_dict
    }
    # TODO: change later from llm=None
    empty_prompt_txt = prompt.format(llm=None, **fmt_dict)
    return empty_prompt_txt


def get_biggest_prompt(prompts: List[Prompt]) -> Prompt:
    """Get biggest prompt.

    Oftentimes we need to fetch the biggest prompt, in order to
    be the most conservative about chunking text. This
    is a helper utility for that.

    """
    empty_prompt_txts = [get_empty_prompt_txt(prompt) for prompt in prompts]
    empty_prompt_txt_lens = [len(txt) for txt in empty_prompt_txts]
    biggest_prompt = prompts[empty_prompt_txt_lens.index(max(empty_prompt_txt_lens))]
    return biggest_prompt
