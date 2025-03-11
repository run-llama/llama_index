from typing import Optional

from playwright.async_api import Browser as AsyncBrowser

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.agentql.const import (
    DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
    DEFAULT_WAIT_FOR_NETWORK_IDLE,
    DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
    DEFAULT_RESPONSE_MODE,
    REQUEST_ORIGIN,
)
from llama_index.tools.agentql.utils import _aget_current_agentql_page


class GetWebElementBrowserSpec(BaseToolSpec):
    """
    Get a web element from a browser.
    """

    spec_functions = [
        "get_web_element_from_browser",
    ]

    def __init__(
        self,
        async_browser: AsyncBrowser,
        timeout: Optional[int] = DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
        wait_for_network_idle: Optional[bool] = DEFAULT_WAIT_FOR_NETWORK_IDLE,
        include_hidden: Optional[bool] = DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
        mode: Optional[str] = DEFAULT_RESPONSE_MODE,
    ):
        """
        Initialize GetWebElementBrowserSpec.

        Args:
            async_browser: An async playwright browser instance.
            timeout: The number of seconds to wait for a request before timing out. Defaults to 300.
            wait_for_network_idle: Whether to wait until the network reaches a full idle state. Defaults to `True`.
            include_hidden: Whether to take into account visually hidden elements on the page. Defaults to `False`.

            mode: 'standard' uses deep data analysis, while 'fast' trades some depth of analysis for speed and is adequate for most usecases.
            Learn more about the modes in this guide: https://docs.agentql.com/accuracy/standard-mode. Defaults to 'fast'.
        """
        self.async_browser = async_browser
        self.timeout = timeout
        self.wait_for_network_idle = wait_for_network_idle
        self.include_hidden = include_hidden
        self.mode = mode

    async def get_web_element_from_browser(
        self,
        prompt: str,
    ) -> str:
        """
        Finds a web element on the active web page in a running browser instance using element’s Natural Language description and returns its CSS selector for further interaction, like clicking, filling a form field, etc.

        Args:
            prompt: Accepts Natural Language description of the web element to find on the page.

        Returns:
            str: The CSS selector of the target element.
        """
        page = await _aget_current_agentql_page(self.async_browser)
        element = await page.get_by_prompt(
            prompt,
            self.timeout,
            self.wait_for_network_idle,
            self.include_hidden,
            self.mode,
            request_origin=REQUEST_ORIGIN,
        )
        tf_id = await element.get_attribute("tf623_id")
        return f"[tf623_id='{tf_id}']"
