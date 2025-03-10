from typing import Optional
import os

from playwright.async_api import Browser as AsyncBrowser

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.agentql.const import (
    DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
    DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
    DEFAULT_IS_STEALTH_MODE_ENABLED,
    DEFAULT_WAIT_FOR_NETWORK_IDLE,
    DEFAULT_INCLUDE_HIDDEN_DATA,
    DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_WAIT_FOR_PAGE_LOAD_SECONDS,
    DEFAULT_IS_SCROLL_TO_BOTTOM_ENABLED,
    DEFAULT_IS_SCREENSHOT_ENABLED,
    DEFAULT_API_TIMEOUT_SECONDS,
    REQUEST_ORIGIN,
)
from llama_index.tools.agentql.messages import (
    QUERY_PROMPT_REQUIRED_ERROR_MESSAGE,
    QUERY_PROMPT_EXCLUSIVE_ERROR_MESSAGE,
    UNSET_API_KEY_ERROR_MESSAGE,
    MISSING_BROWSER_ERROR_MESSAGE,
)
from llama_index.tools.agentql.load_data import aload_data
from llama_index.tools.agentql.utils import (
    _aget_current_agentql_page,
)


class AgentQLToolSpec(BaseToolSpec):
    """
    AgentQL tool spec.
    """

    spec_functions = [
        "extract_web_data_with_rest_api",
        "extract_web_data_from_browser",
        "get_web_element_from_browser",
    ]

    def __init__(
        self,
        async_browser: Optional[AsyncBrowser] = None,
        api_key: Optional[str] = None,
        extract_data_timeout: Optional[int] = DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
        extract_elements_timeout: Optional[
            int
        ] = DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
        api_timeout: Optional[int] = DEFAULT_API_TIMEOUT_SECONDS,
        is_stealth_mode_enabled: Optional[bool] = DEFAULT_IS_STEALTH_MODE_ENABLED,
        wait_for_page_load: Optional[int] = DEFAULT_WAIT_FOR_PAGE_LOAD_SECONDS,
        wait_for_network_idle: Optional[bool] = DEFAULT_WAIT_FOR_NETWORK_IDLE,
        include_hidden_data: Optional[bool] = DEFAULT_INCLUDE_HIDDEN_DATA,
        include_hidden_elements: Optional[bool] = DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
        is_scroll_to_bottom_enabled: Optional[
            bool
        ] = DEFAULT_IS_SCROLL_TO_BOTTOM_ENABLED,
        mode: Optional[str] = DEFAULT_RESPONSE_MODE,
        is_screenshot_enabled: Optional[bool] = DEFAULT_IS_SCREENSHOT_ENABLED,
    ) -> None:
        """
        Initialize AgentQLToolSpec.

        Args:
            async_browser: An async browser instance. Required for extract_web_data_from_browser/get_web_element_from_browser.
            api_key: AgentQL API key. You can create one at https://dev.agentql.com.

            extract_data_timeout: The number of seconds to wait for a request before timing out for **extracting data from browser**. Defaults to 900.
            extract_elements_timeout: The number of seconds to wait for a request before timing out for **getting elements from browser**. Defaults to 300.
            api_timeout: The number of seconds to wait for a request before timing out for **extracting data from rest api**. Defaults to 900.

            is_stealth_mode_enabled: Whether to enable experimental anti-bot evasion strategies. This feature may not work for all websites at all times.
            Data extraction may take longer to complete with this mode enabled. Defaults to `False`.

            wait_for_page_load: The number of seconds to wait for the page to load before **extracting data from rest api**. Defaults to 0.
            wait_for_network_idle: Whether to wait until the network reaches a full idle state before **extracting data/getting elements from browser**. Defaults to `True`.

            include_hidden_data: Whether to take into account visually hidden elements on the page for **extracting data from browser**. Defaults to `False`.
            include_hidden_elements: Whether to take into account visually hidden elements on the page for **getting elements from browser**. Defaults to `False`.

            is_scroll_to_bottom_enabled: Whether to scroll to bottom of the page before **extracting data from rest api**. Defaults to `False`

            mode: 'standard' uses deep data analysis, while 'fast' trades some depth of analysis for speed and is adequate for most usecases.
            Learn more about the modes in this guide: https://docs.agentql.com/accuracy/standard-mode. Defaults to 'fast'.

            is_screenshot_enabled: Whether to take a screenshot before **extracting data from rest api**. Returned in 'metadata' as a Base64 string. Defaults to `False`
        """
        self.async_browser = async_browser

        self._api_key = api_key or os.getenv("AGENTQL_API_KEY")
        if not self._api_key:
            raise ValueError(UNSET_API_KEY_ERROR_MESSAGE)

        self.extract_data_timeout = extract_data_timeout
        self.extract_elements_timeout = extract_elements_timeout
        self.api_timeout = api_timeout

        self.is_stealth_mode_enabled = is_stealth_mode_enabled

        self.wait_for_page_load = wait_for_page_load
        self.wait_for_network_idle = wait_for_network_idle

        self.include_hidden_data = include_hidden_data
        self.include_hidden_elements = include_hidden_elements

        self.is_scroll_to_bottom_enabled = is_scroll_to_bottom_enabled
        self.mode = mode
        self.is_screenshot_enabled = is_screenshot_enabled

    async def extract_web_data_with_rest_api(
        self, url: str, query: Optional[str] = None, prompt: Optional[str] = None
    ) -> dict:
        """
        Extracts structured data as JSON from a web page given a URL using either an AgentQL query or a Natural Language description of the data.

        Args:
            url: Accepts the URL of the public webpage to extract data from.
            query: Accepts AgentQL query used to extract the data. The query must be enclosed with curly braces `{}`. Either this field or `prompt` field must be provided.
            prompt: Accepts Natural Language description of the data to extract from the page. If AgentQL query is not specified, always use the `prompt` field. Either this field or `query` field must be provided.

        Returns:
            dict: The extracted data.
        """
        _params = {
            "wait_for": self.wait_for_page_load,
            "is_scroll_to_bottom_enabled": self.is_scroll_to_bottom_enabled,
            "mode": self.mode,
            "is_screenshot_enabled": self.is_screenshot_enabled,
        }
        _metadata = {
            "experimental_stealth_mode_enabled": self.is_stealth_mode_enabled,
        }

        return await aload_data(
            url=url,
            query=query,
            prompt=prompt,
            params=_params,
            metadata=_metadata,
            api_key=self._api_key,
            timeout=self.api_timeout,
        )

    async def extract_web_data_from_browser(
        self,
        query: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """
        Extracts structured data as a JSON from the active web page in a running browser instance using either an AgentQL query or a Natural Language description of the data.

        Args:
            query: Accepts AgentQL query used to extract the data. The query must be enclosed with curly braces `{}`. Either this field or `prompt` field must be provided.
            prompt: Accepts Natural Language description of the data to extract from the page. If AgentQL query is not specified, always use the `prompt` field. Either this field or `query` field must be provided.

        Returns:
            dict: The extracted data.
        """
        # Check if an async browser instance is provided
        if not self.async_browser:
            raise ValueError(MISSING_BROWSER_ERROR_MESSAGE)

        # Check that query and prompt cannot be both empty or both provided
        if not query and not prompt:
            raise ValueError(QUERY_PROMPT_REQUIRED_ERROR_MESSAGE)
        if query and prompt:
            raise ValueError(QUERY_PROMPT_EXCLUSIVE_ERROR_MESSAGE)

        page = await _aget_current_agentql_page(self.async_browser)
        if query:
            return await page.query_data(
                query,
                self.extract_data_timeout,
                self.wait_for_network_idle,
                self.include_hidden_data,
                self.mode,
                request_origin=REQUEST_ORIGIN,
            )
        else:
            return await page.get_data_by_prompt_experimental(
                prompt,
                self.extract_data_timeout,
                self.wait_for_network_idle,
                self.include_hidden_data,
                self.mode,
                request_origin=REQUEST_ORIGIN,
            )

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
        # Check if an async browser instance is provided
        if not self.async_browser:
            raise ValueError(MISSING_BROWSER_ERROR_MESSAGE)

        page = await _aget_current_agentql_page(self.async_browser)
        element = await page.get_by_prompt(
            prompt,
            self.extract_elements_timeout,
            self.wait_for_network_idle,
            self.include_hidden_elements,
            self.mode,
            request_origin=REQUEST_ORIGIN,
        )
        tf_id = await element.get_attribute("tf623_id")
        return f"[tf623_id='{tf_id}']"
