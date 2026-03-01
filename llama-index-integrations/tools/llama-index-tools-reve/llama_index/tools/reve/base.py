"""
Reve AI image generation tool spec.

Uses the native Reve API at api.reve.com for image creation,
editing, and remixing. Supports test-time scaling, postprocessing
(upscale, background removal, resize), and credit tracking.
"""

import base64
import json
import os
from typing import Any, Dict, Optional

import httpx

from llama_index.core.tools.tool_spec.base import BaseToolSpec

REVE_API_BASE = "https://api.reve.com"

RESPONSE_HEADER_KEYS = [
    "x-credits-used",
    "x-credits-remaining",
    "x-model-version",
    "x-request-id",
    "x-content-violation",
]


class ReveToolSpec(BaseToolSpec):
    """
    Reve AI image generation tool spec.

    Provides tools for image creation, editing, remixing, and credit
    checking using the native Reve API (api.reve.com).

    Args:
        api_key: Reve API key. Falls back to REVE_API_KEY env var.
        base_url: API base URL. Defaults to https://api.reve.com.
        timeout: HTTP request timeout in seconds. Defaults to 120.

    """

    spec_functions = [
        "reve_create_image",
        "reve_edit_image",
        "reve_remix_image",
        "reve_check_credits",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = REVE_API_BASE,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("REVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Reve API key is required. Pass api_key or set REVE_API_KEY env var."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    @staticmethod
    def _parse_response_headers(response: httpx.Response) -> Dict[str, Any]:
        """Extract Reve metadata from response headers."""
        meta: Dict[str, Any] = {}
        for key in RESPONSE_HEADER_KEYS:
            value = response.headers.get(key)
            if value is not None:
                clean_key = key.removeprefix("x-").replace("-", "_")
                meta[clean_key] = value
        return meta

    @staticmethod
    def _build_result(
        response: httpx.Response, body: Dict[str, Any]
    ) -> str:
        """Combine response body with header metadata into a JSON string."""
        meta = ReveToolSpec._parse_response_headers(response)
        result = {**body, "meta": meta}
        return json.dumps(result, indent=2)

    async def _fetch_and_encode_image(self, image_url: str) -> str:
        """Download an image from a URL and return its base64 encoding."""
        client = self._get_client()
        resp = await client.get(image_url)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")

    async def reve_create_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        test_time_scaling: Optional[int] = None,
        upscale_factor: Optional[int] = None,
        remove_background: Optional[bool] = None,
        fit_max_dim: Optional[int] = None,
    ) -> str:
        """
        Generate an image using Reve AI.

        Args:
            prompt: Text description of the image to generate.
            aspect_ratio: Aspect ratio, e.g. '1:1', '16:9', '9:16'.
            test_time_scaling: Quality parameter 1-15. 3-5 for high quality,
                10+ for maximum quality. Higher values use more credits.
            upscale_factor: Postprocessing upscale 2-4x.
            remove_background: If True, remove the image background.
            fit_max_dim: Resize to fit within this max dimension (pixels).

        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
        }
        if test_time_scaling is not None:
            payload["test_time_scaling"] = test_time_scaling
        if upscale_factor is not None:
            payload["upscale_factor"] = upscale_factor
        if remove_background is not None:
            payload["remove_background"] = remove_background
        if fit_max_dim is not None:
            payload["fit_max_dim"] = fit_max_dim

        client = self._get_client()
        response = await client.post("/v1/image/create", json=payload)
        response.raise_for_status()
        return self._build_result(response, response.json())

    async def reve_edit_image(
        self,
        prompt: str,
        image_url: str,
        aspect_ratio: str = "1:1",
        test_time_scaling: Optional[int] = None,
        upscale_factor: Optional[int] = None,
        remove_background: Optional[bool] = None,
        fit_max_dim: Optional[int] = None,
    ) -> str:
        """
        Edit an existing image using Reve AI.

        Downloads the image from image_url, base64-encodes it, and
        sends it to Reve's edit endpoint with the given prompt.

        Args:
            prompt: Text description of the desired edit.
            image_url: URL of the source image to edit.
            aspect_ratio: Aspect ratio, e.g. '1:1', '16:9', '9:16'.
            test_time_scaling: Quality parameter 1-15.
            upscale_factor: Postprocessing upscale 2-4x.
            remove_background: If True, remove the image background.
            fit_max_dim: Resize to fit within this max dimension (pixels).

        """
        image_b64 = await self._fetch_and_encode_image(image_url)

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "image": image_b64,
            "aspect_ratio": aspect_ratio,
        }
        if test_time_scaling is not None:
            payload["test_time_scaling"] = test_time_scaling
        if upscale_factor is not None:
            payload["upscale_factor"] = upscale_factor
        if remove_background is not None:
            payload["remove_background"] = remove_background
        if fit_max_dim is not None:
            payload["fit_max_dim"] = fit_max_dim

        client = self._get_client()
        response = await client.post("/v1/image/edit", json=payload)
        response.raise_for_status()
        return self._build_result(response, response.json())

    async def reve_remix_image(
        self,
        prompt: str,
        image_url: str,
        aspect_ratio: str = "1:1",
        test_time_scaling: Optional[int] = None,
        upscale_factor: Optional[int] = None,
        remove_background: Optional[bool] = None,
        fit_max_dim: Optional[int] = None,
    ) -> str:
        """
        Remix an existing image using Reve AI.

        Downloads the image from image_url, base64-encodes it, and
        sends it to Reve's remix endpoint with the given prompt.

        Args:
            prompt: Text description of the desired remix.
            image_url: URL of the source image to remix.
            aspect_ratio: Aspect ratio, e.g. '1:1', '16:9', '9:16'.
            test_time_scaling: Quality parameter 1-15.
            upscale_factor: Postprocessing upscale 2-4x.
            remove_background: If True, remove the image background.
            fit_max_dim: Resize to fit within this max dimension (pixels).

        """
        image_b64 = await self._fetch_and_encode_image(image_url)

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "image": image_b64,
            "aspect_ratio": aspect_ratio,
        }
        if test_time_scaling is not None:
            payload["test_time_scaling"] = test_time_scaling
        if upscale_factor is not None:
            payload["upscale_factor"] = upscale_factor
        if remove_background is not None:
            payload["remove_background"] = remove_background
        if fit_max_dim is not None:
            payload["fit_max_dim"] = fit_max_dim

        client = self._get_client()
        response = await client.post("/v1/image/remix", json=payload)
        response.raise_for_status()
        return self._build_result(response, response.json())

    async def reve_check_credits(self) -> str:
        """
        Check remaining Reve AI credits.

        Makes a minimal API call to read the credit balance from
        response headers. Uses approximately 1 credit.

        """
        payload: Dict[str, Any] = {
            "prompt": "test",
            "aspect_ratio": "1:1",
        }

        client = self._get_client()
        response = await client.post("/v1/image/create", json=payload)
        response.raise_for_status()
        meta = self._parse_response_headers(response)
        return json.dumps(
            {
                "credits_remaining": meta.get("credits_remaining", "unknown"),
                "credits_used": meta.get("credits_used", "unknown"),
            },
            indent=2,
        )
