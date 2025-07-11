"""
Cloudflare AI Gateway LLM integration.

This module provides integration with Cloudflare AI Gateway, allowing you to
use multiple AI models from different providers with automatic fallback.
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Union
import logging

import httpx
from llama_index.core.llms import LLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.base.llms.types import LLMMetadata

from .providers import get_provider_config

logger = logging.getLogger(__name__)


class CloudflareAIGatewayError(Exception):
    """Base exception for Cloudflare AI Gateway errors."""

    pass


class CloudflareAIGatewayUnauthorizedError(CloudflareAIGatewayError):
    """Raised when AI Gateway authentication fails."""

    pass


class CloudflareAIGatewayDoesNotExistError(CloudflareAIGatewayError):
    """Raised when AI Gateway does not exist."""

    pass


class CloudflareAIGatewayOptions(BaseModel):
    """Options for Cloudflare AI Gateway requests."""

    cache_key: Optional[str] = Field(default=None, description="Custom cache key")
    cache_ttl: Optional[int] = Field(
        default=None, ge=0, description="Cache time-to-live in seconds"
    )
    skip_cache: bool = Field(default=False, description="Bypass caching")
    metadata: Optional[Dict[str, Union[str, int, bool, None]]] = Field(
        default=None, description="Custom metadata for the request"
    )
    collect_log: Optional[bool] = Field(
        default=None, description="Enable/disable log collection"
    )
    event_id: Optional[str] = Field(default=None, description="Custom event identifier")
    request_timeout_ms: Optional[int] = Field(
        default=None, ge=0, description="Request timeout in milliseconds"
    )


class AIGatewayClientWrapper:
    """Wrapper for HTTP clients that intercepts requests and routes through AI Gateway."""

    def __init__(self, gateway_instance, original_client, llm_instance):
        self.gateway = gateway_instance
        self.original_client = original_client
        self.llm = llm_instance
        self.provider_config = get_provider_config(llm_instance)

        if not self.provider_config:
            raise CloudflareAIGatewayError(
                f"Unsupported provider for LLM: {type(self.llm).__name__}"
            )

    def __getattr__(self, name):
        """Delegate attribute access to the original client."""
        return getattr(self.original_client, name)

    def post(self, url: str, **kwargs):
        """Intercept POST requests and route through AI Gateway."""
        # Transform request for AI Gateway
        transformed_request = self._transform_request(url, kwargs)

        # Make request to AI Gateway
        return self.gateway._make_ai_gateway_request(transformed_request)

    def _transform_request(self, url: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request for AI Gateway format."""
        # Extract headers and body
        headers = kwargs.get("headers", {})
        json_data = kwargs.get("json", {})

        # Get endpoint from URL
        endpoint = self.provider_config.transform_endpoint(url)

        # Pass the original request body directly to AI Gateway
        # AI Gateway handles provider-specific format differences internally
        return {
            "provider": self.provider_config.name,
            "endpoint": endpoint,
            "headers": headers,
            "query": json_data,
        }


class CloudflareAIGateway(LLM):
    """
    Cloudflare AI Gateway LLM.

    This class intercepts requests to multiple LLM providers and routes them through
    Cloudflare AI Gateway for automatic fallback and load balancing.

    The key concept is that you provide multiple LLM instances (from different providers),
    and this class intercepts their requests, transforms them for AI Gateway, and
    delegates the actual LLM functionality to the first available provider.

    Args:
        llms: List of LLM instances to use (will be tried in order)
        account_id: Your Cloudflare account ID
        gateway: The name of your AI Gateway
        api_key: Your Cloudflare API key (optional if using binding)
        binding: Cloudflare AI Gateway binding (alternative to account_id/gateway/api_key)
        options: Request-level options for AI Gateway
        max_retries: Maximum number of retries for API calls
        timeout: Timeout for API requests in seconds
        callback_manager: Callback manager for observability
        default_headers: Default headers for API requests
        http_client: Custom httpx client
        async_http_client: Custom async httpx client
    """

    llms: List[LLM] = Field(
        description="List of LLM instances to use (will be tried in order)"
    )
    account_id: Optional[str] = Field(
        default=None, description="Your Cloudflare account ID"
    )
    gateway: Optional[str] = Field(
        default=None, description="The name of your AI Gateway"
    )
    api_key: Optional[str] = Field(default=None, description="Your Cloudflare API key")
    binding: Optional[Any] = Field(
        default=None, description="Cloudflare AI Gateway binding"
    )
    options: Optional[CloudflareAIGatewayOptions] = Field(
        default=None, description="Request-level options for AI Gateway"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries for API calls", ge=0
    )
    timeout: float = Field(
        default=60.0, description="Timeout for API requests in seconds", ge=0
    )
    default_headers: Optional[Dict[str, str]] = Field(
        default=None, description="Default headers for API requests"
    )
    http_client: Optional[httpx.Client] = Field(
        default=None, description="Custom httpx client"
    )
    async_http_client: Optional[httpx.AsyncClient] = Field(
        default=None, description="Custom async httpx client"
    )

    _client: Optional[httpx.Client] = PrivateAttr()
    _aclient: Optional[httpx.AsyncClient] = PrivateAttr()
    _current_llm_index: int = PrivateAttr(default=0)
    _original_clients: Dict[int, Any] = PrivateAttr(default_factory=dict)
    _original_async_clients: Dict[int, Any] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        llms: List[LLM],
        account_id: Optional[str] = None,
        gateway: Optional[str] = None,
        api_key: Optional[str] = None,
        binding: Optional[Any] = None,
        options: Optional[CloudflareAIGatewayOptions] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        **kwargs: Any,
    ) -> None:
        # Validate configuration
        if not llms:
            raise ValueError("At least one LLM must be provided")

        if binding is None:
            if not account_id or not gateway:
                raise ValueError(
                    "Either binding or account_id+gateway must be provided"
                )
            if not api_key:
                raise ValueError("api_key is required when not using binding")

        super().__init__(
            llms=llms,
            account_id=account_id,
            gateway=gateway,
            api_key=api_key,
            binding=binding,
            options=options,
            max_retries=max_retries,
            timeout=timeout,
            callback_manager=callback_manager,
            default_headers=default_headers,
            http_client=http_client,
            async_http_client=async_http_client,
            **kwargs,
        )

        self._client = http_client
        self._aclient = async_http_client

        # Inject AI Gateway client into each LLM
        self._inject_ai_gateway_clients()

    def _inject_ai_gateway_clients(self) -> None:
        """Inject AI Gateway client into each LLM to intercept requests."""
        for i, llm in enumerate(self.llms):
            # Store original client if it exists
            if hasattr(llm, "_client") and llm._client is not None:
                self._original_clients[i] = llm._client
                llm._client = AIGatewayClientWrapper(self, llm._client, llm)

            # Store original async client if it exists
            if hasattr(llm, "_aclient") and llm._aclient is not None:
                self._original_async_clients[i] = llm._aclient
                llm._aclient = AIGatewayClientWrapper(self, llm._aclient, llm)

    def _get_client(self) -> httpx.Client:
        """Get HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers=self.default_headers,
            )
        return self._client

    def _get_aclient(self) -> httpx.AsyncClient:
        """Get async HTTP client."""
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.default_headers,
            )
        return self._aclient

    def _parse_options_to_headers(
        self, options: Optional[CloudflareAIGatewayOptions]
    ) -> Dict[str, str]:
        """Parse options to headers."""
        headers = {}

        if options is None:
            return headers

        if options.skip_cache:
            headers["cf-skip-cache"] = "true"

        if options.cache_ttl is not None:
            headers["cf-cache-ttl"] = str(options.cache_ttl)

        if options.metadata:
            headers["cf-aig-metadata"] = json.dumps(options.metadata)

        if options.collect_log is not None:
            headers["cf-aig-collect-log"] = str(options.collect_log).lower()

        if options.event_id:
            headers["cf-aig-event-id"] = options.event_id

        if options.request_timeout_ms is not None:
            headers["cf-aig-request-timeout-ms"] = str(options.request_timeout_ms)

        return headers

    def _get_current_llm(self) -> LLM:
        """Get the current LLM to use."""
        if not self.llms:
            raise CloudflareAIGatewayError("No LLMs configured")
        return self.llms[self._current_llm_index % len(self.llms)]

    def _try_next_llm(self) -> None:
        """Try the next LLM in the list."""
        self._current_llm_index += 1
        if self._current_llm_index >= len(self.llms):
            raise CloudflareAIGatewayError("All LLMs failed")

    def _make_ai_gateway_request(self, request_body: Dict[str, Any]) -> httpx.Response:
        """Make request to AI Gateway."""
        if self.binding is not None:
            # Use binding - this would need to be implemented based on the binding interface
            raise NotImplementedError("Binding support not yet implemented")
        else:
            # Use API
            headers = self._parse_options_to_headers(self.options)
            headers.update(
                {
                    "Content-Type": "application/json",
                    "cf-aig-authorization": f"Bearer {self.api_key}",
                }
            )

            url = (
                f"https://gateway.ai.cloudflare.com/v1/{self.account_id}/{self.gateway}"
            )

            client = self._get_client()
            response = client.post(url, json=request_body, headers=headers)

            # Handle response
            self._handle_ai_gateway_response(response)

            return response

    def _handle_ai_gateway_response(self, response: httpx.Response) -> None:
        """Handle AI Gateway response and check for errors."""
        if response.status_code == 400:
            try:
                result = response.json()
                if (
                    not result.get("success")
                    and result.get("error")
                    and result["error"][0].get("code") == 2001
                ):
                    raise CloudflareAIGatewayDoesNotExistError(
                        "This AI gateway does not exist"
                    )
            except (ValueError, KeyError, IndexError):
                pass
            raise CloudflareAIGatewayError(f"Bad request: {response.text}")

        elif response.status_code == 401:
            try:
                result = response.json()
                if (
                    not result.get("success")
                    and result.get("error")
                    and result["error"][0].get("code") == 2009
                ):
                    raise CloudflareAIGatewayUnauthorizedError(
                        "Your AI Gateway has authentication active, but you didn't provide a valid apiKey"
                    )
            except (ValueError, KeyError, IndexError):
                pass
            raise CloudflareAIGatewayError("Unauthorized")

        elif response.status_code != 200:
            raise CloudflareAIGatewayError(
                f"Request failed with status {response.status_code}: {response.text}"
            )

    @classmethod
    def class_name(cls) -> str:
        return "CloudflareAIGateway"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata from the current LLM."""
        current_llm = self._get_current_llm()
        return current_llm.metadata

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with the AI Gateway by delegating to the current LLM."""
        while True:
            try:
                current_llm = self._get_current_llm()
                return current_llm.chat(messages, **kwargs)
            except Exception as e:
                # Try next LLM on failure
                logger.warning(
                    f"It seems that the current LLM is not working with the AI Gateway. Error: {e}"
                )
                self._try_next_llm()
                continue

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Stream chat with the AI Gateway by delegating to the current LLM."""
        while True:
            try:
                current_llm = self._get_current_llm()
                return current_llm.stream_chat(messages, **kwargs)
            except Exception as e:
                # Try next LLM on failure
                logger.warning(
                    f"It seems that the current LLM is not working with the AI Gateway. Error: {e}"
                )
                self._try_next_llm()
                continue

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Complete a prompt using the AI Gateway by delegating to the current LLM."""
        while True:
            try:
                current_llm = self._get_current_llm()
                return current_llm.complete(prompt, formatted, **kwargs)
            except Exception as e:
                # Try next LLM on failure
                logger.warning(
                    f"It seems that the current LLM is not working with the AI Gateway. Error: {e}"
                )
                self._try_next_llm()
                continue

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Stream complete a prompt using the AI Gateway by delegating to the current LLM."""
        while True:
            try:
                current_llm = self._get_current_llm()
                return current_llm.stream_complete(prompt, formatted, **kwargs)
            except Exception as e:
                # Try next LLM on failure
                logger.warning(
                    f"It seems that the current LLM is not working with the AI Gateway. Error: {e}"
                )
                self._try_next_llm()
                continue

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat with the AI Gateway by delegating to the current LLM."""
        while True:
            try:
                current_llm = self._get_current_llm()
                return await current_llm.achat(messages, **kwargs)
            except Exception as e:
                # Try next LLM on failure
                logger.warning(
                    f"It seems that the current LLM is not working with the AI Gateway. Error: {e}"
                )
                self._try_next_llm()
                continue

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async stream chat with the AI Gateway by delegating to the current LLM."""
        while True:
            try:
                current_llm = self._get_current_llm()
                return current_llm.astream_chat(messages, **kwargs)
            except Exception as e:
                # Try next LLM on failure
                logger.warning(
                    f"It seems that the current LLM is not working with the AI Gateway. Error: {e}"
                )
                self._try_next_llm()
                continue

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Async complete a prompt using the AI Gateway by delegating to the current LLM."""
        while True:
            try:
                current_llm = self._get_current_llm()
                return await current_llm.acomplete(prompt, formatted, **kwargs)
            except Exception as e:
                # Try next LLM on failure
                logger.warning(
                    f"It seems that the current LLM is not working with the AI Gateway. Error: {e}"
                )
                self._try_next_llm()
                continue

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async stream complete a prompt using the AI Gateway by delegating to the current LLM."""
        while True:
            try:
                current_llm = self._get_current_llm()
                return current_llm.astream_complete(prompt, formatted, **kwargs)
            except Exception as e:
                # Try next LLM on failure
                logger.warning(
                    f"It seems that the current LLM is not working with the AI Gateway. Error: {e}"
                )
                self._try_next_llm()
                continue
