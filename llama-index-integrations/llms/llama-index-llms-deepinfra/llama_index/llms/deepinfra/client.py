import aiohttp
import requests
from typing import Any, Dict, Generator, AsyncGenerator, Optional

from llama_index.llms.deepinfra.utils import (
    retry_request,
    maybe_decode_sse_data,
    aretry_request,
)
from llama_index.llms.deepinfra.constants import API_BASE, USER_AGENT


class DeepInfraClient:
    def __init__(
        self,
        api_key: str,
        api_base: Optional[str] = API_BASE,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = 10,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.max_retries = max_retries

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a synchronous request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            payload (Dict[str, Any]): The request payload.

        Returns:
                Dict[str, Any]: The API response.
        """

        def perform_request():
            response = requests.post(
                self.get_url(endpoint),
                json={
                    **payload,
                    "stream": False,
                },
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        return retry_request(perform_request, max_retries=self.max_retries)

    def request_stream(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """
        Perform a synchronous streaming request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            payload (Dict[str, Any]): The request payload.

        Yields:
            str: The streaming response from the API.
        """

        def perform_request():
            response = requests.post(
                self.get_url(endpoint),
                json={
                    **payload,
                    "stream": True,
                },
                headers=self._get_headers(),
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if resp := maybe_decode_sse_data(line):
                    yield resp

        response = retry_request(perform_request, max_retries=self.max_retries)
        yield from response

    async def arequest(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform an asynchronous request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            payload (Dict[str, Any]): The request payload.

        Returns:
            Dict[str, Any]: The API response.
        """

        async def perform_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.get_url(endpoint),
                    json={
                        **payload,
                        "stream": False,
                    },
                    headers=self._get_headers(),
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    return await response.json()

        return await aretry_request(perform_request, max_retries=self.max_retries)

    async def arequest_stream(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Perform an asynchronous streaming request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            payload (Dict[str, Any]): The request payload.

        Yields:
            str: The streaming response from the API.
        """

        async def perform_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.get_url(endpoint),
                    json={
                        **payload,
                        "stream": True,
                    },
                    headers=self._get_headers(),
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if resp := maybe_decode_sse_data(line):
                            yield resp

        response = await aretry_request(perform_request, max_retries=self.max_retries)
        async for resp in response:
            yield resp

    def get_url(self, endpoint: str) -> str:
        """
        Get DeepInfra API URL.
        """
        return f"{self.api_base}/{endpoint}"

    def get_model_details(self, model_name: str) -> requests.Response:
        """
        Get model details from DeepInfra API.
        If the model does not exist, a 404 response is returned.

        Returns:
            requests.Response: The API response.
        """
        request_url = self.get_url(f"models/{model_name}")
        return requests.get(request_url, headers=self._get_headers())

    def is_function_calling_model(self, model_name: str) -> bool:
        """
        Check if the model is a function calling model.

        Returns:
            bool: True if the model is a function calling model, False otherwise.
        """
        response = self.get_model_details(model_name)
        if response.status_code == 404:
            return False
        response_json = response.json()
        tags = response_json.get("tags", [])
        return "tools" in tags
