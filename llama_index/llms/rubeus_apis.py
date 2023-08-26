from typing import Optional, Union, List, Dict, Any
import httpx

from llama_index.llms.rubeus_utils import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    ProviderTypes,
    RubeusCacheType,
    ProviderBase,
    RubeusModes,
    RetrySettings,
    Message
)

from .rubeus_client import APIClient

__all__ = ["Completions", "ChatCompletions"]


class APIResource:
    _client: APIClient

    def __init__(self, client: APIClient) -> None:
        self._client = client
        # self._get = client.get
        self._post = client.post
        # self._patch = client.patch
        # self._put = client.put
        # self._delete = client.delete
        # self._get_api_list = client.get_api_list


class Completions(APIResource):
    def create(
        self,
        *,
        prompt: str = "",
        timeout: Union[float, None] = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        provider: Optional[ProviderTypes] = ProviderTypes.OPENAI,
        model: str = "gpt-3.5-turbo",
        model_api_key: Optional[str] = None,
        temperature: float = 0.1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: List[str] = None,
        stream: Optional[bool] = False,
        max_tokens: Optional[int] = None,
        trace_id: Optional[str] = "",
        cache_status: Optional[RubeusCacheType] = "",
        cache: Optional[bool] = False,
        metadata: Optional[Dict[str, Any]] = None,
        weight: Optional[float] = 1.0,
    ) -> httpx.Response:
        return self._post(
            "/v1/complete",
            body=[
                {
                    "prompt": prompt,
                    "provider": provider,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "model": model,
                    "model_api_key": model_api_key,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "stop_sequences": stop_sequences,
                    "stream": stream,
                    "max_tokens": max_tokens,
                    "trace_id": trace_id,
                    "cache_status": cache_status,
                    "cache": cache,
                    "metadata": metadata or {},
                    "weight": weight,
                }
            ],
            stream=stream,
            mode=RubeusModes.SINGLE.value,
        )

    def with_fallbacks(self, llms: List[ProviderBase]) -> httpx.Response:
        body = []
        for i in llms:
            body.append(i.json())
        return self._post(
            "/v1/chatComplete", body=body, mode=RubeusModes.FALLBACK, stream=False
        )

    def with_loadbalancing(self, llms: List[ProviderBase]) -> httpx.Response:
        body = []
        for i in llms:
            body.append(i.json())
        return self._post(
            "/v1/chatComplete", body=body, mode=RubeusModes.LOADBALANCE, stream=False
        )


class ChatCompletions(APIResource):
    def create(
        self,
        *,
        messages: Message = [],
        timeout: Union[float, None] = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        provider: Optional[ProviderTypes] = ProviderTypes.OPENAI,
        model: str = "gpt-3.5-turbo",
        model_api_key: Optional[str] = None,
        temperature: float = 0.1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: List[str] = None,
        stream: Optional[bool] = False,
        max_tokens: Optional[int] = None,
        trace_id: Optional[str] = "",
        cache_status: Optional[RubeusCacheType] = "",
        cache: Optional[bool] = False,
        metadata: Optional[Dict[str, Any]] = None,
        weight: Optional[float] = 1.0,
        retry_settings: Optional[RetrySettings] = None,
    ) -> httpx.Response:
        return self._post(
            "/v1/chatComplete",
            body=[
                {
                    "messages": messages,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "provider": provider,
                    "model": model,
                    "model_api_key": model_api_key,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "stop_sequences": stop_sequences,
                    "stream": stream,
                    "max_tokens": max_tokens,
                    "trace_id": trace_id,
                    "cache_status": cache_status,
                    "cache": cache,
                    "metadata": metadata or {},
                    "weight": weight,
                    "retry_settings": retry_settings or {},
                }
            ],
            stream=stream,
            mode=RubeusModes.SINGLE.value,
        )

    def with_fallbacks(self, llms: List[ProviderBase]) -> httpx.Response:
        body = []
        for i in llms:
            body.append(i.json())
        return self._post("/v1/chatComplete", body=body, mode=RubeusModes.FALLBACK, stream=False)

    def with_loadbalancing(self, llms: List[ProviderBase]) -> httpx.Response:
        body = []
        for i in llms:
            body.append(i.json())
        return self._post("/v1/chatComplete", body=body, mode=RubeusModes.LOADBALANCE)
