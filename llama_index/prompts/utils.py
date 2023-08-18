from string import Formatter
from typing import List

from llama_index.llms.base import LLM
from llama_index.prompts.base import BasePromptTemplate


def get_empty_prompt_txt(prompt: BasePromptTemplate) -> str:
    """Get empty prompt text.

    Substitute empty strings in parts of the prompt that have
    not yet been filled out. Skip variables that have already
    been partially formatted. This is used to compute the initial tokens.

    """
    fmt_dict = {v: "" for v in prompt.template_vars}
    empty_prompt_txt = prompt.format(llm=None, **fmt_dict)
    return empty_prompt_txt


def get_biggest_prompt(prompts: List[BasePromptTemplate]) -> BasePromptTemplate:
    """Get biggest prompt.

    Oftentimes we need to fetch the biggest prompt, in order to
    be the most conservative about chunking text. This
    is a helper utility for that.

    """
    empty_prompt_txts = [get_empty_prompt_txt(prompt) for prompt in prompts]
    empty_prompt_txt_lens = [len(txt) for txt in empty_prompt_txts]
    biggest_prompt = prompts[empty_prompt_txt_lens.index(max(empty_prompt_txt_lens))]
    return biggest_prompt


def get_template_vars(template_str: str) -> List[str]:
    """Get template variables from a template string."""
    variables = []
    formatter = Formatter()

    for _, variable_name, _, _ in formatter.parse(template_str):
        if variable_name:
            variables.append(variable_name)

    return variables


def is_chat_model(llm: LLM) -> bool:
    return llm.metadata.is_chat_model
