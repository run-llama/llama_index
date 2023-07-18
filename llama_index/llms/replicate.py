from typing import Any, Dict, Optional

from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import (CompletionResponse, CompletionResponseGen,
                                   LLMMetadata)
from llama_index.llms.custom import CustomLLM


class Replicate(CustomLLM):
    def __init__(
        self,
        model: str,
        temperature: float = 0.75,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        prompt_key: str = "prompt",
    ) -> None:
        self._model = model
        self._context_window = context_window
        self._prompt_key = prompt_key

        # model kwargs
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._additional_kwargs = additional_kwargs or {}

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self._context_window, num_output=self._max_tokens
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self._temperature,
            "max_length": self._max_tokens,
        }
        model_kwargs = {
            **base_kwargs,
            **self._additional_kwargs,
        }
        return model_kwargs

    def _get_input_dict(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {self._prompt_key: prompt, **self._model_kwargs, **kwargs}

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response_gen = self.stream_complete(prompt, **kwargs)
        response_list = list(response_gen)
        final_response = response_list[-1]
        final_response.delta = None
        return final_response

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            import replicate
        except ImportError:
            raise ImportError(
                "Could not import replicate library."
                "Please install replicate with `pip install replicate`"
            )

        input_dict = self._get_input_dict(prompt, **kwargs)
        response_iter = replicate.run(self._model, input=input_dict)

        def gen() -> CompletionResponseGen:
            text = ""
            for delta in response_iter:
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                )

        return gen()
