import logging
from threading import Thread
from typing import Any, List, Optional

from llama_index.llms.base import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_callback,
)
from llama_index.llms.custom import CustomLLM
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.prompts.prompts import SimpleInputPrompt

logger = logging.getLogger(__name__)


class HuggingFaceLLM(CustomLLM):
    """HuggingFace LLM."""

    def __init__(
        self,
        context_window: int = 4096,
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
    ) -> None:
        """Initialize params."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        model_kwargs = model_kwargs or {}
        self._model_name = model_name
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, **model_kwargs
        )

        # check context_window
        config_dict = self.model.config.to_dict()
        model_context_window = int(
            config_dict.get("max_position_embeddings", context_window)
        )
        if model_context_window and model_context_window < context_window:
            logger.warning(
                f"Supplied context_window {context_window} is greater "
                "than the model's max input size {model_context_window}. "
                "Disable this warning by setting a lower context_window."
            )
            context_window = model_context_window

        tokenizer_kwargs = tokenizer_kwargs or {}
        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = context_window

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_name, **tokenizer_kwargs
        )

        self._context_window = context_window
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

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self._max_new_tokens,
            model_name=self._model_name,
        )

    @llm_callback(is_chat=False)
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint."""

        full_prompt = self._query_wrapper_prompt.format(query_str=prompt)
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

        return CompletionResponse(text=completion, raw={"model_output": tokens})

    @llm_callback(is_chat=False)
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Streaming completion endpoint."""
        from transformers import TextIteratorStreamer

        full_prompt = self._query_wrapper_prompt.format(query_str=prompt)
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
        def gen() -> CompletionResponseGen:
            text = ""
            for x in streamer:
                text += x
                yield CompletionResponse(text=text, delta=x)

        return gen()
