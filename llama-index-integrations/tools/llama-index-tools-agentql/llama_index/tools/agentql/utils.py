from typing import Optional
import httpx
import agentql
from urllib.parse import urlparse

from playwright.async_api import Browser as AsyncBrowser
from playwright.async_api import Page as AsyncPage

from llama_index.tools.agentql.endpoint import EXTRACT_DATA_ENDPOINT


from llama_index.tools.agentql.message import (
    UNAUTHORIZED_ERROR_MESSAGE,
    QUERY_PROMPT_VALIDATION_ERROR_MESSAGE,
)


async def _aget_current_agentql_page(browser: AsyncBrowser) -> AsyncPage:
    """
    Get the current page of the async browser.

    Args:
        browser: The browser to get the current page from.

    Returns:
        Page: The current page.
    """
    context = browser.contexts[0] if browser.contexts else await browser.new_context()
    page = context.pages[-1] if context.pages else await context.new_page()
    return await agentql.wrap_async(page)


def handle_http_error(e: httpx.HTTPStatusError) -> None:
    response = e.response
    if response.status_code == httpx.codes.UNAUTHORIZED:
        raise ValueError(UNAUTHORIZED_ERROR_MESSAGE) from e

    msg = response.text
    try:
        error_json = response.json()
        msg = (
            error_json["error_info"] if "error_info" in error_json else str(error_json)
        )
    except (ValueError, TypeError):
        msg = f"HTTP {e}."
    raise ValueError(msg) from e


async def aload_data(
    url: str,
    api_key: str,
    metadata: dict,
    params: dict,
    timeout: int,
    query: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    if not query and not prompt:
        raise ValueError(QUERY_PROMPT_VALIDATION_ERROR_MESSAGE)

    payload = {
        "url": url,
        "query": query,
        "prompt": prompt,
        "params": params,
        "metadata": metadata,
    }

    headers = {
        "X-API-Key": f"{api_key}",
        "Content-Type": "application/json",
        "X-TF-Request-Origin": "langchain",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                EXTRACT_DATA_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            handle_http_error(e)
        else:
            return response.json()


def validate_url_scheme(url: str) -> None:
    """Check that the URL scheme is valid."""
    if url:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError("URL scheme must be 'http' or 'https'")
