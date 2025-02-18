from typing import Optional
import httpx
import agentql
from urllib.parse import urlparse

from playwright.async_api import Browser as AsyncBrowser
from playwright.async_api import Page as AsyncPage

# from llama_index.tools.playwright import _aget_current_page
from llama_index.tools.agentql.const import EXTRACT_DATA_ENDPOINT, DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS

async def _aget_current_agentql_page(browser: AsyncBrowser) -> AsyncPage:
    """
    Get the current page of the async browser.

    Args:
        browser: The browser to get the current page from.

    Returns:
        AsyncPage: The current page.
    """
    if not browser.contexts:
        context = await browser.new_context()
        return await agentql.wrap_async(await context.new_page())
    context = browser.contexts[
        0
    ]  # Assuming you're using the default browser context
    if not context.pages:
        return await agentql.wrap_async(await context.new_page())
    # Assuming the last page in the list is the active one
    return await agentql.wrap_async(context.pages[-1])


async def aload_data(
    url: str,
    api_key: str,
    params: dict,
    metadata: dict,
    request_origin: str,
    timeout: int = DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
    query: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    if not query and not prompt:
        raise ValueError("'query' and 'prompt' cannot both be empty")

    payload = {"url": url, "query": query, "prompt": prompt, "params": params, "metadata": metadata}

    headers = {
        "X-API-Key": f"{api_key}",
        "Content-Type": "application/json",
        "X-TF-Request-Origin": request_origin,
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
            response = e.response
            if response.status_code in [401, 403]:
                raise ValueError(
                    "Please, provide a valid API Key. You can create one at https://dev.agentql.com."
                ) from e
            else:
                try:
                    error_json = response.json()
                    msg = (
                        error_json["error_info"]
                        if "error_info" in error_json
                        else str(error_json)
                    )
                except (ValueError, TypeError):
                    msg = f"HTTP {e}."
                raise ValueError(msg) from e
        else:
            data = response.json()
            return data
        
def validate_url_scheme(url: str) -> None:
    """Check that the URL scheme is valid."""
    if url:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError("URL scheme must be 'http' or 'https'")
        
