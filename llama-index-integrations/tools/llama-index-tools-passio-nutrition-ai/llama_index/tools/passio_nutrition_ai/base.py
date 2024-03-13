"""Passio Nutrition Search tool spec."""

from typing import final, NoReturn
from datetime import datetime, timedelta

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

ENDPOINT_BASE_URL = "https://api.passiolife.com/v2/products/napi/food/search/advanced"


class NoDiskStorage:
    @final
    def __getstate__(self) -> NoReturn:
        raise AttributeError("Do not store on disk.")

    @final
    def __setstate__(self, state) -> NoReturn:
        raise AttributeError("Do not store on disk.")


try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_random,
        wait_exponential,
        retry_if_result,
    )
except ImportError:
    # No retries if tenacity is not installed.
    def retry(f, *args, **kwargs):
        return f

    def stop_after_attempt(n):
        return None

    def wait_random(a, b):
        return None

    def wait_exponential(multiplier, min, max):
        return None


def is_http_retryable(rsp):
    # -return rsp and rsp.status_code >= 500
    return (
        rsp
        and not isinstance(rsp, dict)
        and rsp.status_code in [408, 425, 429, 500, 502, 503, 504]
    )


class ManagedPassioLifeAuth(NoDiskStorage):
    """Manages the token for the NutritionAI API."""

    def __init__(self, subscription_key: str):
        self.subscription_key = subscription_key
        self._last_token = None
        self._access_token_expiry = None
        self._access_token = None
        self._customer_id = None

    @property
    def headers(self) -> dict:
        if not self.is_valid_now():
            self.refresh_access_token()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Passio-ID": self._customer_id,
        }

    def is_valid_now(self):
        return (
            self._access_token is not None
            and self._customer_id is not None
            and self._access_token_expiry is not None
            and self._access_token_expiry > datetime.now()
        )

    @retry(
        retry=retry_if_result(is_http_retryable),
        stop=stop_after_attempt(4),
        wait=wait_random(0, 0.3) + wait_exponential(multiplier=1, min=0.1, max=2),
    )
    def _http_get(self, subscription_key):
        return requests.get(
            f"https://api.passiolife.com/v2/token-cache/napi/oauth/token/{subscription_key}"
        )

    def refresh_access_token(self):
        """Refresh the access token for the NutritionAI API."""
        rsp = self._http_get(self.subscription_key)
        if not rsp:
            raise ValueError("Could not get access token")
        self._last_token = token = rsp.json()
        self._customer_id = token["customer_id"]
        self._access_token = token["access_token"]
        self._access_token_expiry = (
            datetime.now()
            + timedelta(seconds=token["expires_in"])
            - timedelta(seconds=5)
        )  # 5 seconds: approximate time for a token refresh to be processed.


class NutritionAIToolSpec(BaseToolSpec):
    """Tool that queries the Passio Nutrition AI API."""

    spec_functions = ["nutrition_ai_search"]
    auth_: ManagedPassioLifeAuth

    def __init__(self, api_key: str) -> None:
        """Initialize with parameters."""
        self.auth_ = ManagedPassioLifeAuth(api_key)

    @retry(
        retry=retry_if_result(is_http_retryable),
        stop=stop_after_attempt(4),
        wait=wait_random(0, 0.3) + wait_exponential(multiplier=1, min=0.1, max=2),
    )
    def _http_get(self, query: str):
        return requests.get(
            ENDPOINT_BASE_URL,
            headers=self.auth_.headers,
            params={"term": query},  # type: ignore
        )

    def _nutrition_request(self, query: str):
        response = self._http_get(query)
        if not response:
            raise ValueError("No response from NutritionAI API.")
        return response.json()

    def nutrition_ai_search(self, query: str):
        """
        Retrieve nutrition facts for a given food item.
        Input should be a search query string for the food item.

        Args:
            query (str): The food item to look for.

        Returns a JSON result with the nutrition facts for the food item and, if available, alternative food items which sometimes are a better match.
        """
        return self._nutrition_request(query)
