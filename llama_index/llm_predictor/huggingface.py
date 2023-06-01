"""Huggingface LLM Wrapper."""

import logging
from threading import Thread
from typing import Any, List, Generator, Optional, Tuple

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType
from llama_index.llm_predictor.base import BaseLLMPredictor, LLMMetadata
from llama_index.prompts.base import Prompt
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.prompts.prompts import SimpleInputPrompt

logger = logging.getLogger(__name__)


class HuggingFaceLLMPredictor(BaseLLMPredictor):
    """Huggingface Specific LLM predictor class.

    Wrapper around an LLMPredictor to provide streamlined access to HuggingFace models.

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
        max_input_size: int = 4096,
        max_new_tokens: int = 256,
        system_prompt: str = "",
        query_wrapper_prompt: SimpleInputPrompt = DEFAULT_SIMPLE_INPUT_PROMPT,
        tokenizer_name: str = "StabilityAI/stablelm-tuned-alpha-3b",
        model_name: str = "StabilityAI/stablelm-tuned-alpha-3b",
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device_map: str = "auto",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = None,
        tokenizer_outputs_to_remove: Optional[list] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initialize params."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        self.callback_manager = callback_manager or CallbackManager([])

        model_kwargs = model_kwargs or {}
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, **model_kwargs
        )

        # check max_input_size
        config_dict = self.model.config.to_dict()
        model_max_input_size = int(
            config_dict.get("max_position_embeddings", max_input_size)
        )
        if model_max_input_size and model_max_input_size < max_input_size:
            logger.warning(
                f"Supplied max_input_size {max_input_size} is greater "
                "than the model's max input size {model_max_input_size}. "
                "Disable this warning by setting a lower max_input_size."
            )
            max_input_size = model_max_input_size

        tokenizer_kwargs = tokenizer_kwargs or {}
        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = max_input_size

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_name, **tokenizer_kwargs
        )

        self._max_input_size = max_input_size
        self._max_new_tokens = max_new_tokens

        self._generate_kwargs = generate_kwargs or {}
        self._device_map = device_map
        self._tokenizer_outputs_to_remove = tokenizer_outputs_to_remove or []
        self._system_prompt = system_prompt
        self._query_wrapper_prompt = query_wrapper_prompt
        self._total_tokens_used = 0
        self._last_token_usage: Optional[int] = None

        # setup stopping criteria
        stopping_ids_list = stopping_ids or []

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs: Any,
            ) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        self._stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    def get_llm_metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self._max_input_size, num_output=self._max_new_tokens
        )

    def stream(self, prompt: Prompt, **prompt_args: Any) -> Tuple[Generator, str]:
        """Stream the answer to a query.

        NOTE: this is a beta feature. Will try to build or use
        better abstractions about response handling.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            str: The predicted answer.

        """
        from transformers import TextIteratorStreamer

        formatted_prompt = prompt.format(**prompt_args)
        full_prompt = self._query_wrapper_prompt.format(query_str=formatted_prompt)
        if self._system_prompt:
            full_prompt = f"{self._system_prompt} {full_prompt}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in self._tokenizer_outputs_to_remove:
            if key in inputs:
                inputs.pop(key, None)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            decode_kwargs={"skip_special_tokens": True},
        )
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=self._max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            **self._generate_kwargs,
        )

        # generate in background thread
        # NOTE/TODO: token counting doesn't work with streaming
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # create generator based off of streamer
        def response() -> Generator:
            for x in streamer:
                yield x

        return response(), formatted_prompt

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

        llm_payload = {**prompt_args}
        llm_payload["template"] = prompt
        event_id = self.callback_manager.on_event_start(
            CBEventType.LLM, payload=llm_payload
        )

        formatted_prompt = prompt.format(**prompt_args)
        full_prompt = self._query_wrapper_prompt.format(query_str=formatted_prompt)
        if self._system_prompt:
            full_prompt = f"{self._system_prompt} {full_prompt}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in self._tokenizer_outputs_to_remove:
            if key in inputs:
                inputs.pop(key, None)

        tokens = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            **self._generate_kwargs,
        )
        completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
        self._total_tokens_used += len(completion_tokens) + inputs["input_ids"].size(1)
        completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)

        self.callback_manager.on_event_end(
            CBEventType.LLM,
            payload={"response": completion, "formatted_prompt": formatted_prompt},
            event_id=event_id,
        )
        return completion, formatted_prompt

    async def apredict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Async predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        return self.predict(prompt, **prompt_args)
