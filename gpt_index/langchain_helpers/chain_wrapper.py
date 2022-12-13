"""Wrapper functions around an LLM chain."""

from typing import Any, Optional, Tuple

from langchain import LLMChain, OpenAI
from langchain.llms.base import LLM

from gpt_index.prompts.base import Prompt


class LLMPredictor:
    """LLM predictor class.

    Wrapper around an LLMChain from Langchain.

    Args:
        llm (Optional[LLM]): LLM from Langchain to use for predictions.
            Defaults to OpenAI's text-davinci-002 model.
            Please see
            `Langchain's LLM Page
            <https://langchain.readthedocs.io/en/latest/modules/llms.html>`_
            for more details.

    """

    def __init__(self, llm: Optional[LLM] = None) -> None:
        """Initialize params."""
        self._llm = llm or OpenAI(temperature=0, model_name="text-davinci-002")

    def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        llm_chain = LLMChain(prompt=prompt, llm=self._llm)

        formatted_prompt = prompt.format(**prompt_args)
        full_prompt_args = prompt.get_full_format_args(prompt_args)
        return llm_chain.predict(**full_prompt_args), formatted_prompt
