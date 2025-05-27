from typing import Optional
import agentql
import httpx

from llama_index.tools.agentql.const import EXTRACT_DATA_ENDPOINT, REQUEST_ORIGIN
from llama_index.tools.agentql.messages import (
    QUERY_PROMPT_REQUIRED_ERROR_MESSAGE,
    QUERY_PROMPT_EXCLUSIVE_ERROR_MESSAGE,
    UNAUTHORIZED_ERROR_MESSAGE,
)

try:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import Page as AsyncPage
except ImportError as e:
    raise ImportError(
        "Unable to import playwright. Please make sure playwright module is properly installed."
    ) from e


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


def _handle_http_error(e: httpx.HTTPStatusError) -> None:
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


async def _aload_data(
    url: str,
    api_key: str,
    metadata: dict,
    params: dict,
    timeout: int,
    query: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    if not query and not prompt:
        raise ValueError(QUERY_PROMPT_REQUIRED_ERROR_MESSAGE)
    if query and prompt:
        raise ValueError(QUERY_PROMPT_EXCLUSIVE_ERROR_MESSAGE)

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
        "X-TF-Request-Origin": REQUEST_ORIGIN,
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
            _handle_http_error(e)
        else:
            return response.json()
