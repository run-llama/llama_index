"""AI21 Contextual Answers."""

from gpt_index.llm_predictor.base import BaseLLMPredictor, LLMMetadata
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from typing import Any, Tuple, Optional, Generator
import requests
import os


class AI21ContextualAnswersPredictor(BaseLLMPredictor):
    """ChatGPT contextual answers class.

    NOTE: this is a beta class, may change.

    Link: https://docs.ai21.com/reference/contextual-answers-api-ref

    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Init params."""
        self._api_key = api_key or os.getenv("AI21_API_KEY")
        self._url = "https://api.ai21.com/studio/v1/experimental/answer"
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
        raise NotImplementedError("Streaming is not supported for this LLM.")

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
        if isinstance(prompt, QuestionAnswerPrompt):
            prompt = prompt.partial_format(**prompt_args)
            context = prompt.partial_dict["context_str"]
            query = prompt.partial_dict["query_str"]
        elif isinstance(prompt, RefinePrompt):
            prompt = prompt.partial_format(**prompt_args)
            context = (
                f"Previous answer to the question: {prompt.partial_dict['existing_answer']}\n"
                f"New Context: {prompt.partial_dict['context_msg']}\n"
            )
            query = prompt.partial_dict["query_str"]
        else:
            raise ValueError("Prompt must be a QuestionAnswerPrompt or RefinePrompt.")

        formatted_prompt = f"{context}\nQuestion: {query}\nAnswer:"

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        json = {
            "context": context,
            "question": query,
        }

        response = requests.post(self._url, headers=headers, json=json)
        response_json = response.json()
        if "answer" not in response_json:
            raise ValueError(
                f"Response does not contain an answer. Response: {response_json}"
            )
        return response_json["answer"], formatted_prompt

    async def apredict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Async predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        return self.predict(prompt, **prompt_args)
