"""Rubeus implementation."""
import os
from typing import Optional, Union, Mapping
from httpx import Timeout
from llama_index.llms.rubeus_utils import (
    MISSING_API_KEY_ERROR_MESSAGE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    Params,
)
from llama_index.llms.rubeus_client import APIClient
from . import rubeus_apis

__all__ = ["Rubeus"]


class Rubeus(APIClient):
    completion = rubeus_apis.Completions
    chat_completion = rubeus_apis.ChatCompletions

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, Timeout, None] = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Optional[Mapping[str, object]] = None,
        default_params: Params = None,
    ) -> None:
        if base_url is None:
            self.base_url = "https://api.portkey.ai"
        self.api_key = api_key or os.environ.get("PORTKEY_API_KEY", "")
        if not self.api_key:
            raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)
        self._default_params = default_params or {}
        self._timeout = timeout
        self._max_retries = max_retries
        print('default_headers: ', self._default_params)
        self._default_headers = default_headers
        self._default_query = default_query
        super().__init__(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
            custom_headers=self._default_headers,
            custom_query=self._default_query,
            custom_params=self._default_params,
        )
        self.completion = rubeus_apis.Completions(self)
        self.chat_completion = rubeus_apis.ChatCompletions(self)
