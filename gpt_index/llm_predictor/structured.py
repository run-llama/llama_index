"""Structured LLM Predictor."""


import logging
from typing import Any, Generator, Tuple

from gpt_index.llm_predictor.base import LLMPredictor
from gpt_index.prompts.base import Prompt

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
            # TODO: return other formats
            parsed_llm_prediction = str(prompt.output_parser.parse(llm_prediction))
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
            parsed_llm_prediction = str(prompt.output_parser.parse(llm_prediction))
        else:
            parsed_llm_prediction = llm_prediction
        return parsed_llm_prediction, formatted_prompt
