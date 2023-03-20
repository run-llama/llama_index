"""Structured LLM Predictor."""


import logging
from dataclasses import dataclass
from typing import Any, Generator, Optional, Tuple

import openai
from langchain import Cohere, LLMChain, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import AI21
from langchain.schema import BaseLanguageModel

# from gpt_index.llm_predictor.base import BaseLLMPredictor, LLMMetadata, LLMPredictor
from gpt_index.llm_predictor.base import LLMPredictor
from gpt_index.prompts.base import Prompt
from gpt_index.utils import (
    ErrorToRetry,
    globals_helper,
    retry_on_exceptions_with_backoff,
)

logger = logging.getLogger(__name__)


class StructuredLLMPredictor(LLMPredictor):
    """Structured LLM predictor class.

    Args:
        llm_predictor (BaseLLMPredictor): LLM Predictor to use.

    """

    def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        llm_prediction, formatted_prompt = super().predict(prompt, **prompt_args)
        # run output parser
        if prompt.output_parser is not None:
            parsed_llm_prediction = prompt.output_parser.parse(llm_prediction)
        else:
            parsed_llm_prediction = llm_prediction

        return parsed_llm_prediction, formatted_prompt

    def stream(self, prompt: Prompt, **prompt_args: Any) -> Tuple[Generator, str]:
        """Stream the answer to a query.

        NOTE: this is a beta feature. Will try to build or use
        better abstractions about response handling.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            str: The predicted answer.

        """
        raise NotImplementedError(
            "Streaming is not supported for structured LLM predictor."
        )

    async def apredict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Async predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        llm_prediction, formatted_prompt = await super().apredict(prompt, **prompt_args)
        if prompt.output_parser is not None:
            parsed_llm_prediction = prompt.output_parser.parse(llm_prediction)
        else:
            parsed_llm_prediction = llm_prediction
        return parsed_llm_prediction, formatted_prompt


# class StructuredLLMPredictor(BaseLLMPredictor):
#     """Structured LLM predictor class.

#     Takes in an LLMPredictor and an output parser.

#     Args:
#         llm_predictor (BaseLLMPredictor): LLM Predictor to use.

#     """

#     def __init__(self, llm_predictor: BaseLLMPredictor) -> None:
#         """Initialize params."""
#         self.llm_predictor = llm_predictor

#     def get_llm_metadata(self) -> LLMMetadata:
#         """Get LLM metadata."""
#         return self.llm_predictor.get_llm_metadata()

#     def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
#         """Predict the answer to a query.

#         Args:
#             prompt (Prompt): Prompt to use for prediction.

#         Returns:
#             Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

#         """
#         llm_prediction, formatted_prompt = self.llm_predictor.predict(
#             prompt, **prompt_args
#         )
#         # run output parser
#         parsed_llm_prediction = prompt.output_parser.parse(llm_prediction)

#         return parsed_llm_prediction, formatted_prompt

#     def stream(self, prompt: Prompt, **prompt_args: Any) -> Tuple[Generator, str]:
#         """Stream the answer to a query.

#         NOTE: this is a beta feature. Will try to build or use
#         better abstractions about response handling.

#         Args:
#             prompt (Prompt): Prompt to use for prediction.

#         Returns:
#             str: The predicted answer.

#         """
#         return self.llm_predictor.stream(prompt, **prompt_args)

#     @property
#     def total_tokens_used(self) -> int:
#         """Get the total tokens used so far."""
#         return self.llm_predictor.total_tokens_used

#     @property
#     def last_token_usage(self) -> int:
#         """Get the last token usage."""
#         return self.llm_predictor.last_token_usage

#     @last_token_usage.setter
#     def last_token_usage(self, value: int) -> None:
#         """Set the last token usage."""
#         self.llm_predictor.last_token_usage = value

#     async def apredict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
#         """Async predict the answer to a query.

#         Args:
#             prompt (Prompt): Prompt to use for prediction.

#         Returns:
#             Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

#         """
#         # run output parser
#         llm_prediction, formatted_prompt = await self.llm_predictor.apredict(
#             prompt, **prompt_args
#         )
#         parsed_llm_prediction = prompt.output_parser.parse(llm_prediction)
#         return parsed_llm_prediction, formatted_prompt
