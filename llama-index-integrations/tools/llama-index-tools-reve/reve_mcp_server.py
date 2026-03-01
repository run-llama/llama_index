"""
Standalone FastMCP server for Reve AI image generation.

Run directly:
    REVE_API_KEY=your-key python reve_mcp_server.py

Or configure in Claude Desktop / Cursor:
    {
      "mcpServers": {
        "reve_mcp": {
          "command": "python3",
          "args": ["/path/to/reve_mcp_server.py"],
          "env": {"REVE_API_KEY": "your-key"}
        }
      }
    }
"""

import base64
import json
import os
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import FastMCP

REVE_API_BASE = "https://api.reve.com"

RESPONSE_HEADER_KEYS = [
    "x-credits-used",
    "x-credits-remaining",
    "x-model-version",
    "x-request-id",
    "x-content-violation",
]

mcp = FastMCP("reve-image-gen")

_client: Optional[httpx.AsyncClient] = None


def _get_api_key() -> str:
    key = os.environ.get("REVE_API_KEY")
    if not key:
        raise ValueError("REVE_API_KEY environment variable is required.")
    return key


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=REVE_API_BASE,
            headers={
                "Authorization": f"Bearer {_get_api_key()}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
    return _client


def _parse_headers(response: httpx.Response) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key in RESPONSE_HEADER_KEYS:
        value = response.headers.get(key)
        if value is not None:
            clean_key = key.removeprefix("x-").replace("-", "_")
            meta[clean_key] = value
    return meta


def _build_result(response: httpx.Response, body: Dict[str, Any]) -> str:
    meta = _parse_headers(response)
    return json.dumps({**body, "meta": meta}, indent=2)


async def _fetch_and_encode_image(image_url: str) -> str:
    client = _get_client()
    resp = await client.get(image_url)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


@mcp.tool()
async def reve_create_image(
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
        aspect_ratio: e.g. '1:1', '16:9', '9:16'.
        test_time_scaling: Quality 1-15. 3-5 high, 10+ max. More credits.
        upscale_factor: Postprocessing upscale 2-4x.
        remove_background: Remove the image background.
        fit_max_dim: Resize to fit within this max dimension (pixels).

    """
    payload: Dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect_ratio}
    if test_time_scaling is not None:
        payload["test_time_scaling"] = test_time_scaling
    if upscale_factor is not None:
        payload["upscale_factor"] = upscale_factor
    if remove_background is not None:
        payload["remove_background"] = remove_background
    if fit_max_dim is not None:
        payload["fit_max_dim"] = fit_max_dim

    client = _get_client()
    response = await client.post("/v1/image/create", json=payload)
    response.raise_for_status()
    return _build_result(response, response.json())


@mcp.tool()
async def reve_edit_image(
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

    Downloads the image from image_url, base64-encodes it, and sends
    it to the Reve edit endpoint.

    Args:
        prompt: Description of the desired edit.
        image_url: URL of the source image to edit.
        aspect_ratio: e.g. '1:1', '16:9', '9:16'.
        test_time_scaling: Quality 1-15.
        upscale_factor: Postprocessing upscale 2-4x.
        remove_background: Remove the image background.
        fit_max_dim: Resize to fit within this max dimension (pixels).

    """
    image_b64 = await _fetch_and_encode_image(image_url)
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

    client = _get_client()
    response = await client.post("/v1/image/edit", json=payload)
    response.raise_for_status()
    return _build_result(response, response.json())


@mcp.tool()
async def reve_remix_image(
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

    Downloads the image from image_url, base64-encodes it, and sends
    it to the Reve remix endpoint.

    Args:
        prompt: Description of the desired remix.
        image_url: URL of the source image to remix.
        aspect_ratio: e.g. '1:1', '16:9', '9:16'.
        test_time_scaling: Quality 1-15.
        upscale_factor: Postprocessing upscale 2-4x.
        remove_background: Remove the image background.
        fit_max_dim: Resize to fit within this max dimension (pixels).

    """
    image_b64 = await _fetch_and_encode_image(image_url)
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

    client = _get_client()
    response = await client.post("/v1/image/remix", json=payload)
    response.raise_for_status()
    return _build_result(response, response.json())


@mcp.tool()
async def reve_check_credits() -> str:
    """
    Check remaining Reve AI credits.

    Makes a minimal API call to read the credit balance from response
    headers. Uses approximately 1 credit.
    """
    payload: Dict[str, Any] = {"prompt": "test", "aspect_ratio": "1:1"}
    client = _get_client()
    response = await client.post("/v1/image/create", json=payload)
    response.raise_for_status()
    meta = _parse_headers(response)
    return json.dumps(
        {
            "credits_remaining": meta.get("credits_remaining", "unknown"),
            "credits_used": meta.get("credits_used", "unknown"),
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
