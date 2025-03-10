import agentql

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
