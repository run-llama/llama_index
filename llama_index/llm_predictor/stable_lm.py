"""Stable LM.

NOTE: this is a beta wrapper, will replace once better abstractions
(e.g. from langchain) come out.

"""
from typing import Any, Generator, Optional, Tuple

from llama_index.llm_predictor.base import BaseLLMPredictor, LLMMetadata
from llama_index.prompts.base import Prompt

DEFAULT_SYSTEM_PROMPT = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""  # noqa: E501


class StableLMPredictor(BaseLLMPredictor):
    """ChatGPT Specific LLM predictor class.

    Wrapper around an LLMPredictor to provide ChatGPT specific features.

    Args:
        llm (Optional[langchain.llms.base.LLM]): LLM from Langchain to use
            for predictions. Defaults to OpenAI's text-davinci-003 model.
            Please see `Langchain's LLM Page
            <https://langchain.readthedocs.io/en/latest/modules/llms.html>`_
            for more details.

        retry_on_throttling (bool): Whether to retry on rate limit errors.
            Defaults to true.

    """

    def __init__(
        self,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tokenizer_name: str = "StabilityAI/stablelm-tuned-alpha-3b",
        model_name: str = "StabilityAI/stablelm-tuned-alpha-3b",
    ) -> None:
        """Initialize params."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.half().cuda()

        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._do_sample = do_sample

        self._system_prompt = system_prompt
        self._total_tokens_used = 0
        self._last_token_usage: Optional[int] = None

    def get_llm_metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata()

    def stream(self, prompt: Prompt, **prompt_args: Any) -> Tuple[Generator, str]:
        """Stream the answer to a query.

        NOTE: this is a beta feature. Will try to build or use
        better abstractions about response handling.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            str: The predicted answer.

        """
        raise NotImplementedError("Streaming is not supported for StableLM.")

    @property
    def total_tokens_used(self) -> int:
        """Get the total tokens used so far."""
        return self._total_tokens_used

    @property
    def last_token_usage(self) -> int:
        """Get the last token usage."""
        return self._last_token_usage or 0

    @last_token_usage.setter
    def last_token_usage(self, value: int) -> None:
        """Set the last token usage."""
        self._last_token_usage = value

    def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        from transformers import StoppingCriteriaList

        formatted_prompt = prompt.format(**prompt_args)
        full_prompt = (
            f"{self._system_prompt}" f"<|USER|>{formatted_prompt}<|ASSISTANT|>"
        )
        input_tokens = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
        tokens = self.model.generate(
            **input_tokens,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            do_sample=self._do_sample,
            stopping_criteria=StoppingCriteriaList(),
        )
        completion_tokens = tokens[0][input_tokens["input_ids"].size(1) :]
        completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return completion, formatted_prompt

    async def apredict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Async predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        return self.predict(prompt, **prompt_args)
