"""Wrapper functions around an LLM chain."""

from typing import Any, Optional, Tuple

from langchain import LLMChain, OpenAI
from langchain.llms.base import LLM

from gpt_index.prompts.base import Prompt


class LLMPredictor:
    """LLM predictor class."""

    def __init__(self, llm: Optional[LLM] = None) -> None:
        """Initialize params."""
        self._llm = llm or OpenAI(temperature=0, model_name="text-davinci-002")

    def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Predict the answer to a query."""
        llm_chain = LLMChain(prompt=prompt, llm=self._llm)

        formatted_prompt = prompt.format(**prompt_args)
        full_prompt_args = prompt.get_full_format_args(prompt_args)
        return llm_chain.predict(**full_prompt_args), formatted_prompt
