import json
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)


class Ollama(CustomLLM):
    base_url: str = Field(description="Base url the model is hosted under.")
    model: str = Field(description="The Ollama model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    prompt_key: str = Field(
        default="prompt", description="The key to use for the prompt in API calls."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Ollama API."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.75,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        prompt_key: str = "prompt",
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)

        super().__init__(
            model=model,
            temperature=temperature,
            base_url=base_url,
            additional_kwargs=additional_kwargs or {},
            context_window=context_window,
            prompt_key=prompt_key,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Ollama_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_length": self.context_window,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def _get_input_dict(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {self.prompt_key: prompt, **self._model_kwargs, **kwargs}

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self._messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response_gen = self.stream_complete(prompt, **kwargs)
        response_list = list(response_gen)
        final_response = response_list[-1]
        final_response.delta = None
        return final_response

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Could not import requests library."
                "Please install requests with `pip install requests`"
            )
        all_kwargs = self._get_all_kwargs(**kwargs)
        prompt = self._completion_to_prompt(prompt)
        response = requests.post(
            url=f"{self.base_url}/api/generate/",
            headers={"Content-Type": "application/json"},
            json={"prompt": prompt, "model": self.model, **all_kwargs},
            stream=True,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            optional_detail = response.json().get("error")
            raise ValueError(
                f"Ollama call failed with status code {response.status_code}."
                f" Details: {optional_detail}"
            )

        def gen(response_iter: Iterator[Any]) -> CompletionResponseGen:
            text = ""
            for stream_response in response_iter:
                delta = json.loads(stream_response).get("response", "")
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                )

        return gen(response.iter_lines(decode_unicode=True))
