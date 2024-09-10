import asyncio
import time
from typing import Dict, Any, List, Sequence

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

try:
    from alibabacloud_searchplat20240529.models import (
        GetTextGenerationRequest,
        GetTextGenerationResponse,
        GetTextGenerationRequestMessages,
    )
    from alibabacloud_tea_openapi.models import Config as AISearchConfig
    from alibabacloud_searchplat20240529.client import Client
    from alibabacloud_tea_util.models import RuntimeOptions
    from Tea.exceptions import TeaException
except ImportError:
    raise ImportError(
        "Could not import alibabacloud_searchplat20240529 python package. "
        "Please install it with `pip install alibabacloud_searchplat20240529`."
    )


def retry_decorator(func, wait_seconds: int = 1):
    def wrap(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except TeaException as e:
                if e.code == "Throttling.RateQuota":
                    time.sleep(wait_seconds)
                else:
                    raise

    return wrap


def aretry_decorator(func, wait_seconds: int = 1):
    async def wrap(*args, **kwargs):
        while True:
            try:
                return await func(*args, **kwargs)
            except TeaException as e:
                if e.code == "Throttling.RateQuota":
                    await asyncio.sleep(wait_seconds)
                else:
                    raise

    return wrap


class AlibabaCloudAISearchLLM(CustomLLM):
    """
    For further details, please visit `https://help.aliyun.com/zh/open-search/search-platform/developer-reference/text-generation-api-details`.
    """

    _client: Client = PrivateAttr()
    _options: RuntimeOptions = PrivateAttr()

    aisearch_api_key: str = Field(default=None, exclude=True)
    endpoint: str = None

    service_id: str = "ops-qwen-turbo"
    workspace_name: str = "default"

    temperature: float = 0.5
    top_k: float = 1
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

    read_timeout: int = 60000
    connection_timeout: int = 5000
    csi_level: str = "strict"

    def __init__(
        self, endpoint: str = None, aisearch_api_key: str = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.aisearch_api_key = get_from_param_or_env(
            "aisearch_api_key", aisearch_api_key, "AISEARCH_API_KEY"
        )
        self.endpoint = get_from_param_or_env("endpoint", endpoint, "AISEARCH_ENDPOINT")

        config = AISearchConfig(
            bearer_token=self.aisearch_api_key,
            endpoint=self.endpoint,
            protocol="http",
        )

        self._client = Client(config=config)

        self._options = RuntimeOptions(
            read_timeout=self.read_timeout, connect_timeout=self.connection_timeout
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(model_name=self.service_id, is_chat_model=True)

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            **self.additional_kwargs,
        }

    @staticmethod
    def _convert_chat_messages(
        messages: Sequence[ChatMessage],
    ) -> List[GetTextGenerationRequestMessages]:
        results = []
        for message in messages:
            message = GetTextGenerationRequestMessages(
                content=message.content, role=message.role
            )
            results.append(message)
        return results

    @retry_decorator
    def _get_text_generation(
        self, messages: List[GetTextGenerationRequestMessages], **kwargs: Any
    ) -> GetTextGenerationResponse:
        parameters: Dict[str, Any] = self._default_params
        parameters.update(kwargs)
        request = GetTextGenerationRequest(
            csi_level=self.csi_level, messages=messages, parameters=parameters
        )

        response: GetTextGenerationResponse = (
            self._client.get_text_generation_with_options(
                workspace_name=self.workspace_name,
                service_id=self.service_id,
                request=request,
                headers={},
                runtime=self._options,
            )
        )
        return response

    @aretry_decorator
    async def _aget_text_generation(
        self, messages: List[GetTextGenerationRequestMessages], **kwargs: Any
    ) -> GetTextGenerationResponse:
        parameters: Dict[str, Any] = self._default_params
        parameters.update(kwargs)
        request = GetTextGenerationRequest(
            csi_level=self.csi_level, messages=messages, parameters=parameters
        )

        response: GetTextGenerationResponse = (
            await self._client.get_text_generation_with_options_async(
                workspace_name=self.workspace_name,
                service_id=self.service_id,
                request=request,
                headers={},
                runtime=self._options,
            )
        )

        return response

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [
            GetTextGenerationRequestMessages(content=prompt, role=MessageRole.USER)
        ]
        response: GetTextGenerationResponse = self._get_text_generation(
            messages, **kwargs
        )
        text = response.body.result.text
        return CompletionResponse(text=text, raw=response)

    def stream_complete(self, messages: Any, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [
            GetTextGenerationRequestMessages(content=prompt, role=MessageRole.USER)
        ]
        response: GetTextGenerationResponse = await self._aget_text_generation(
            messages, **kwargs
        )
        text = response.body.result.text
        return CompletionResponse(text=text, raw=response)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages = self._convert_chat_messages(messages)
        response: GetTextGenerationResponse = self._get_text_generation(
            messages, **kwargs
        )
        text = response.body.result.text
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=response
        )

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        messages = self._convert_chat_messages(messages)
        response: GetTextGenerationResponse = await self._aget_text_generation(
            messages, **kwargs
        )
        text = response.body.result.text
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=response
        )

    @classmethod
    def class_name(cls) -> str:
        return "AlibabaCloudAISearchLLM"
