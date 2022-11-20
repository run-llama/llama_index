"""Wrapper functions around an LLM chain."""

from typing import Any, Dict, Optional, Tuple

from langchain import LLMChain, OpenAI

from gpt_index.prompts.base import Prompt


def openai_llm_predict(
    prompt: Prompt, llm_args_dict: Optional[Dict] = None, **prompt_args: Any
) -> Tuple[str, str]:
    """Predict using OpenAI LLM with a prompt string.

    Also return the formatted prompt.

    """
    llm_args_dict = llm_args_dict or {}
    llm = OpenAI(temperature=0, **llm_args_dict)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    formatted_prompt = prompt.format(**prompt_args)
    full_prompt_args = prompt.get_full_format_args(prompt_args)
    return llm_chain.predict(**full_prompt_args), formatted_prompt
