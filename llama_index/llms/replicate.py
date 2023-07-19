from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import completion_response_to_chat_response
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.llms.generic_utils import stream_completion_response_to_chat_response


class Replicate(CustomLLM):
    def __init__(
        self,
        model: str,
        temperature: float = 0.75,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        prompt_key: str = "prompt",
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
    ) -> None:
        self._model = model
        self._context_window = context_window
        self._prompt_key = prompt_key
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)

        # model kwargs
        self._temperature = temperature
        self._additional_kwargs = additional_kwargs or {}

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self._context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self._model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self._temperature,
            "max_length": self._context_window,
        }
        model_kwargs = {
            **base_kwargs,
            **self._additional_kwargs,
        }
        return model_kwargs

    def _get_input_dict(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {self._prompt_key: prompt, **self._model_kwargs, **kwargs}

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self._messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

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

        prompt = self._completion_to_prompt(prompt)
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
