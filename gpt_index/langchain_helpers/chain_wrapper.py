"""Wrapper functions around an LLM chain."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from langchain import Cohere, LLMChain, OpenAI
from langchain.llms import AI21
from langchain.llms.base import BaseLLM

from gpt_index.constants import MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.prompts.base import Prompt
from gpt_index.utils import globals_helper


@dataclass
class LLMMetadata:
    """LLM metadata.

    We extract this metadata to help with our prompts.

    """

    max_input_size: int = MAX_CHUNK_SIZE
    num_output: int = NUM_OUTPUTS


def _get_llm_metadata(llm: BaseLLM) -> LLMMetadata:
    """Get LLM metadata from llm."""
    if not isinstance(llm, BaseLLM):
        raise ValueError("llm must be an instance of langchain.llms.base.LLM")
    if isinstance(llm, OpenAI):
        return LLMMetadata(
            max_input_size=llm.modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens,
        )
    elif isinstance(llm, Cohere):
        # TODO: figure out max input size for cohere
        return LLMMetadata(num_output=llm.max_tokens)
    elif isinstance(llm, AI21):
        # TODO: figure out max input size for AI21
        return LLMMetadata(num_output=llm.maxTokens)
    else:
        return LLMMetadata()


class LLMPredictor:
    """LLM predictor class.

    Wrapper around an LLMChain from Langchain.

    Args:
        llm (Optional[langchain.llms.base.LLM]): LLM from Langchain to use
            for predictions. Defaults to OpenAI's text-davinci-003 model.
            Please see `Langchain's LLM Page
            <https://langchain.readthedocs.io/en/latest/modules/llms.html>`_
            for more details.

    """

    def __init__(self, llm: Optional[BaseLLM] = None) -> None:
        """Initialize params."""
        self._llm = llm or OpenAI(temperature=0, model_name="text-davinci-003")
        self._total_tokens_used = 0
        self.flag = True
        self._last_token_usage: Optional[int] = None

    def get_llm_metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # TODO: refactor mocks in unit tests, this is a stopgap solution
        if hasattr(self, "_llm"):
            return _get_llm_metadata(self._llm)
        else:
            return LLMMetadata()

    def _predict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Inner predict function."""
        llm_chain = LLMChain(prompt=prompt.get_langchain_prompt(), llm=self._llm)

        # Note: we don't pass formatted_prompt to llm_chain.predict because
        # langchain does the same formatting under the hood
        full_prompt_args = prompt.get_full_format_args(prompt_args)
        llm_prediction = llm_chain.predict(**full_prompt_args)
        return llm_prediction

    def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        formatted_prompt = prompt.format(**prompt_args)
        llm_prediction = self._predict(prompt, **prompt_args)

        # We assume that the value of formatted_prompt is exactly the thing
        # eventually sent to OpenAI, or whatever LLM downstream
        prompt_tokens_count = self._count_tokens(formatted_prompt)
        prediction_tokens_count = self._count_tokens(llm_prediction)
        self._total_tokens_used += prompt_tokens_count + prediction_tokens_count
        return llm_prediction, formatted_prompt

    @property
    def total_tokens_used(self) -> int:
        """Get the total tokens used so far."""
        return self._total_tokens_used

    def _count_tokens(self, text: str) -> int:
        tokens = globals_helper.tokenizer(text)
        return len(tokens)

    @property
    def last_token_usage(self) -> int:
        """Get the last token usage."""
        if self._last_token_usage is None:
            return 0
        return self._last_token_usage

    @last_token_usage.setter
    def last_token_usage(self, value: int) -> None:
        """Set the last token usage."""
        self._last_token_usage = value
