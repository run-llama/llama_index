"""Structured LLM Predictor."""


import logging
from typing import Any

from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.base import Prompt
from llama_index.types import TokenGen

logger = logging.getLogger(__name__)


class StructuredLLMPredictor(LLMPredictor):
    """Structured LLM predictor class.

    Args:
        llm_predictor (BaseLLMPredictor): LLM Predictor to use.

    """

    def predict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        llm_prediction = super().predict(prompt, **prompt_args)
        # run output parser
        if prompt.metadata["output_parser"] is not None:
            # TODO: return other formats
            output_parser = prompt.metadata["output_parser"]
            parsed_llm_prediction = str(output_parser.parse(llm_prediction))
        else:
            parsed_llm_prediction = llm_prediction

        return parsed_llm_prediction

    def stream(self, prompt: Prompt, **prompt_args: Any) -> TokenGen:
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

    async def apredict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Async predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        llm_prediction = await super().apredict(prompt, **prompt_args)
        if prompt.metadata["output_parser"] is not None:
            output_parser = prompt.metadata["output_parser"]
            parsed_llm_prediction = str(output_parser.parse(llm_prediction))
        else:
            parsed_llm_prediction = llm_prediction
        return parsed_llm_prediction
