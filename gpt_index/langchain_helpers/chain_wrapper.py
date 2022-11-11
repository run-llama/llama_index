"""Wrapper functions around an LLM chain."""

from typing import Any, Dict, Optional, Tuple

from langchain import LLMChain, OpenAI, Prompt


def openai_llm_predict(
    prompt_template: str, llm_args_dict: Optional[Dict] = None, **prompt_args: Any
) -> Tuple[str, str]:
    """Predict using OpenAI LLM with a prompt string.

    Also return the formatted prompt.

    """
    llm_args_dict = llm_args_dict or {}
    llm = OpenAI(temperature=0, **llm_args_dict)
    prompt = Prompt(template=prompt_template, input_variables=list(prompt_args.keys()))
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    formatted_prompt = prompt_template.format(**prompt_args)
    return llm_chain.predict(**prompt_args), formatted_prompt
