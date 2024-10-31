import warnings
from typing import Any, Dict, List, Optional, Sequence
import requests

from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.core.schema import ImageNode
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

from llama_index.multi_modal_llms.nvidia.utils import (
    BASE_URL,
    KNOWN_URLS,
    NVIDIA_MULTI_MODAL_MODELS,
    generate_nvidia_multi_modal_chat_message,
    aggregate_msgs,
    process_response,
)


class NVIDIAClient:
    def __init__(
        self,
        api_key: str,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key
        self.timeout = timeout

    def _get_headers(self, stream: bool) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "User-Agent": "langchain-nvidia-ai-endpoints",
        }
        headers["accept"] = "text/event-stream" if stream else "application/json"
        return headers

    def get_model_details(self) -> List[str]:
        """
        Get model details.

        Returns:
            List of models
        """
        return list(NVIDIA_MULTI_MODAL_MODELS.keys())

    def request(
        self,
        endpoint: str,
        stream: bool,
        messages: Dict[str, Any],
        extra_headers: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform a synchronous request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            messages (Dict[str, Any]): The request payload.

        Returns:
            Dict[str, Any]: The API response.
        """

        def perform_request():
            payload = {"messages": messages, "stream": stream, **kwargs}
            headers = {
                **self._get_headers(stream=stream),
                **extra_headers,
            }
            response = requests.post(
                endpoint, json=payload, headers=headers, stream=stream
            )
            response.raise_for_status()
            return response

        return perform_request()


class NVIDIAMultiModal(MultiModalLLM):
    model: str = Field(description="The Multi-Modal model to use from NVIDIA.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: Optional[int] = Field(
        description=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
        gt=0,
    )
    context_window: Optional[int] = Field(
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout, in seconds, for API requests.",
        ge=0,
    )
    api_key: str = Field(default=None, description="The NVIDIA API key.", exclude=True)
    base_url: str = Field(default=BASE_URL, description="The base URL for NVIDIA API.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the NVIDIA API."
    )
    default_headers: Optional[Dict[str, str]] = Field(
        default=None, description="The default headers for API requests."
    )

    def __init__(
        self,
        model: str = "microsoft/phi-3-vision-128k-instruct",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 300,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = BASE_URL,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        is_hosted = base_url in KNOWN_URLS

        if is_hosted and api_key == "NO_API_KEY_PROVIDED":
            warnings.warn(
                "An API key is required for the hosted NIM. This will become an error in 0.2.0.",
            )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=base_url,
            callback_manager=callback_manager,
            default_headers=default_headers,
            **kwargs,
        )

    @property
    def _client(self) -> NVIDIAClient:
        _client = NVIDIAClient(**self._get_credential_kwargs())
        _client._custom_headers = {"User-Agent": "llama-index-multi-modal-llms-nvidia"}
        return _client

    @classmethod
    def class_name(cls) -> str:
        return "nvidia_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            num_output=self.max_tokens or DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    @property
    def available_models(self):
        self._client.get_model_details()

    def _get_credential_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        credential_kwargs = {
            "api_key": self.api_key,
            **kwargs,
        }

        if self.default_headers:
            credential_kwargs["default_headers"] = self.default_headers

        return credential_kwargs

    def _get_multi_modal_chat_messages(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[ImageNode],
    ) -> List[Dict]:
        return generate_nvidia_multi_modal_chat_message(
            prompt=prompt,
            role=role,
            image_documents=image_documents,
        )

    # Model Params for NVIDIA Multi Modal model.
    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        if self.model not in NVIDIA_MULTI_MODAL_MODELS:
            raise ValueError(
                f"Invalid model {self.model}. "
                f"Available models are: {list(NVIDIA_MULTI_MODAL_MODELS.keys())}"
            )
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_tokens is not None:
            base_kwargs["max_tokens"] = self.max_tokens
        return {**base_kwargs, **self.additional_kwargs}

    def _get_response_token_counts(self, raw_response: Any) -> dict:
        """Get the token usage reported by the response."""
        if not isinstance(raw_response, dict):
            return {}

        usage = raw_response.get("usage", {})
        # NOTE: other model providers that use the NVIDIA client may not report usage
        if usage is None:
            return {}

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def _complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict, extra_headers = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        response = self._client.request(
            endpoint=NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
            stream=False,
            messages=message_dict,
            extra_headers=extra_headers,
            **all_kwargs,
        )
        response = response.json()
        text = response["choices"][0]["message"]["content"]
        return CompletionResponse(
            text=text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict, extra_headers = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        def gen() -> CompletionResponseGen:
            response = self._client.request(
                messages=message_dict,
                stream=True,
                endpoint=NVIDIA_MULTI_MODAL_MODELS[self.model]["endpoint"],
                extra_headers=extra_headers,
                **all_kwargs,
            )
            for line in response.iter_lines():
                if line and line.strip() != b"data: [DONE]":
                    line = line.decode("utf-8")
                    line = line[5:]

                    msg, final_line = aggregate_msgs(process_response(line))

                    yield CompletionResponse(
                        **msg,
                        additional_kwargs=self._get_response_token_counts(line),
                    )

                    if final_line:
                        break

        return gen()

    def complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt, image_documents, **kwargs)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, image_documents, **kwargs)

    def chat(
        self,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    def stream_chat(
        self,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    # ===== Async Endpoints =====

    async def _acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError("This function is not yet implemented.")

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        return await self._acomplete(prompt, image_documents, **kwargs)

    async def _astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("This function is not yet implemented.")

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self._astream_complete(prompt, image_documents, **kwargs)

    async def achat(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    async def astream_chat(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This function is not yet implemented.")
