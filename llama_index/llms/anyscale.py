from typing import Any, Dict, Optional, Sequence

import requests

try:
    from pydantic.v1 import Field, PrivateAttr
except ImportError:
    from pydantic import Field, PrivateAttr

import importlib

from llama_index.callbacks import CallbackManager
from llama_index.llms import anyscale_utils

importlib.reload(anyscale_utils)
from llama_index.llms.anyscale_utils import (anyscale_modelname_to_contextsize,
                                             get_from_param_or_env,
                                             messages_to_anyscale_prompt)
from llama_index.llms.base import (LLM, ChatMessage, ChatResponse,
                                   ChatResponseAsyncGen, ChatResponseGen,
                                   CompletionResponse,
                                   CompletionResponseAsyncGen,
                                   CompletionResponseGen, LLMMetadata,
                                   MessageRole, llm_chat_callback,
                                   llm_completion_callback)
from llama_index.llms.generic_utils import (
    achat_to_completion_decorator, astream_chat_to_completion_decorator,
    chat_to_completion_decorator, stream_chat_to_completion_decorator)


class Anyscale(LLM):
    model_name: str = Field(description="The anyscale model to use.")
    api_base: Optional[str] = Field(default=None, description="The base URL to use.")
    api_key: Optional[str] = Field(default=None, description="The base URL to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additonal kwargs for the anyscale API."
    )

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-70b-chat-hf",
        temperature: float = 0.1,
        api_base: Optional[str] = "https://console.endpoints.anyscale.com/m/v1",
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        try:
            import requests
        except ImportError as e:
            raise ImportError(
                "You must install the `requests` package to use Anyscale Endpoint."
                "Please `pip install requests`"
            ) from e
        
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])
        
        api_base = get_from_param_or_env("api_base", api_base, "ANYSCALE_API_BASE")
        api_key  = get_from_param_or_env("api_key",  api_key,  "ANYSCALE_API_KEY")

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Anyscale_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=anyscale_modelname_to_contextsize(self.model_name),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model_name,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        model_kwargs = {
            **base_kwargs,
            **self.additional_kwargs,
        }
        return model_kwargs

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        all_kwargs = self._get_all_kwargs(**kwargs)
        prompt = messages_to_anyscale_prompt(messages)
        body = {
            "messages": prompt,
            "stream": False,
            **all_kwargs
        }

        response = requests.post(url, headers=headers, json=body).json()
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response["choices"][0]["message"]["content"]
            ),
            raw=dict(response["choices"][0]),
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError(
            "Anyscale does not support stream completion in LlamaIndex currently."
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
         raise NotImplementedError(
            "Anyscale does not support Async completion in LlamaIndex currently."
        )

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError(
            "Anyscale does not support Async completion in LlamaIndex currently."
        )

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)
