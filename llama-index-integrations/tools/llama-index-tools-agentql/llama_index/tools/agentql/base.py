from typing import Optional, List
import os

from playwright.async_api import Browser as AsyncBrowser
from playwright.async_api import Page as AsyncPage

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.agentql.const import (
    DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
    DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
    DEFAULT_WAIT_FOR_NETWORK_IDLE,
    DEFAULT_INCLUDE_HIDDEN_DATA,
    DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
    DEFAULT_RESPONSE_MODE,
)
from llama_index.tools.agentql.utils import (
    aload_data,
    _aget_current_agentql_page,
    validate_url_scheme,
)

class AgentQLToolSpec(BaseToolSpec):
    """
    AgentQL tool spec.
    """

    spec_functions = [
        "extract_web_data",
        "extract_web_data_with_browser",
        "extract_web_element_with_browser",
    ]

    def __init__(
        self,
        async_browser: Optional[AsyncBrowser] = None,
        extract_data_timeout: Optional[int] = DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
        extract_elements_timeout: Optional[int] = DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
        wait_for_page_load: Optional[int] = 0,
        wait_for_network_idle: Optional[bool] = DEFAULT_WAIT_FOR_NETWORK_IDLE,
        include_hidden_data: Optional[bool] = DEFAULT_INCLUDE_HIDDEN_DATA,
        include_hidden_elements: Optional[bool] = DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
        response_mode: Optional[str] = DEFAULT_RESPONSE_MODE,
        stealth_mode: Optional[bool] = False,
        is_scroll_to_bottom_enabled: Optional[bool] = False,
        is_screenshot_enabled: Optional[bool] = False,
    ) -> None:
        """
        Initialize AgentQLToolSpec.

        Args:
            async_browser: A browser instance to use for automation.
            extract_data_timeout: Timeout value in seconds for the connection with Extract Data service.
            extract_elements_timeout: Timeout value in seconds for the connection with Extract Elements service.
            wait_for_page_load: Wait time in seconds for page load completion.
            wait_for_network_idle: Whether to wait for network reaching full idle state before querying the page.
            include_hidden_data: Whether to include hidden elements on the page for extract data.
            include_hidden_elements: Whether to include hidden elements on the page for extract elements.
            response_mode: The mode of the query. It can be either 'standard' or 'fast'.
            stealth_mode: Whether to run the browser in stealth mode. This is useful for avoiding detection by anti-bot services (extract_web_data only).
            is_scroll_to_bottom_enabled: Enable scrolling to bottom of the page before extracting data (extract_web_data only).
            is_screenshot_enabled: Whether to take a screenshot of the page before extracting data (extract_web_data only).
        """
        self.async_browser = async_browser

        self.extract_data_timeout = extract_data_timeout
        self.extract_elements_timeout = extract_elements_timeout

        self.wait_for_page_load = wait_for_page_load
        self.wait_for_network_idle = wait_for_network_idle

        self.include_hidden_data = include_hidden_data
        self.include_hidden_elements = include_hidden_elements

        self.response_mode = response_mode
        self.stealth_mode = stealth_mode

        self.is_scroll_to_bottom_enabled = is_scroll_to_bottom_enabled
        self.is_screenshot_enabled = is_screenshot_enabled

    @classmethod
    def from_async_browser(cls, async_browser: AsyncBrowser) -> "AgentQLToolSpec":
        """
        Initialize AgentQLToolSpec from an async browser instance.
        """
        return cls(async_browser=async_browser)
    
    @staticmethod
    async def create_async_playwright_browser(
        headless: bool = True, args: Optional[List[str]] = None
    ) -> AsyncBrowser:
        """
        Create an async playwright browser.

        Args:
            headless: Whether to run the browser in headless mode. Defaults to True.
            args: arguments to pass to browser.chromium.launch

        Returns:
            AsyncBrowser: The playwright browser.
        """
        from playwright.async_api import async_playwright

        browser = await async_playwright().start()
        return await browser.chromium.launch(headless=headless, args=args)
    
    async def extract_web_data(
        self, 
        url: str,
        query: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> dict:
        """
        Extract structure data from a web page given an URL. 
        The data is extracted with either a agentql query or a description of the data to extract.

        Args:
            url: The URL of the web page to extract data from.
            query: The AgentQL query used to extract the data. The query must be enclosed with curly braces `{}`. Either this field or prompt field must be provided. 
            prompt: The natural language description of the data you want to extract. Either this field or query field must be provided.

        Returns:
            dict: The extracted data.
        """
        # Check that the URL scheme is valid
        validate_url_scheme(url)

        # Check if one of 'query' or 'prompt' is provided
        if not query and not prompt:
            raise ValueError("Either 'query' or 'prompt' must be provided")
        
        params = {
            "wait_for": self.wait_for_page_load,
            "is_scroll_to_bottom_enabled": self.is_scroll_to_bottom_enabled,
            "mode": self.response_mode,
            "is_screenshot_enabled": self.is_screenshot_enabled,
        }
        metadata = {
            "experimental_stealth_mode_enabled": self.stealth_mode,
        }
        api_key = os.getenv("AGENTQL_API_KEY")
        if not api_key:
            raise ValueError(
                "AGENTQL_API_KEY environment variable not found. Please set your API key"
            )
        return await aload_data(
            url=url,
            query=query,
            prompt=prompt,
            params=params,
            metadata=metadata,
            api_key=api_key,
            timeout=self.extract_data_timeout,
            request_origin="llamaindex-extractwebdata-tool",
        )
    
    async def extract_web_data_with_browser(
        self,
        query: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """
        Extract structure data from the current web page on a running browser instance. 
        The data is extracted with either a agentql query or a description of the data to extract.

        Args:
            query: The AgentQL query used to extract the data. The query must be enclosed with curly braces `{}`. Either this field or prompt field must be provided. 
            prompt: The natural language description of the data you want to extract. Either this field or query field must be provided.

        Returns:
            dict: The extracted data.
        """
        # Check if an async browser instance is provided
        if not self.async_browser:
            raise ValueError("An async browser instance is required to use this tool.")
        
        # Check if one of 'query' or 'prompt' is provided
        if not query and not prompt:
            raise ValueError("Either 'query' or 'prompt' must be provided")
        
        page = await _aget_current_agentql_page(self.async_browser)
        if query:
            return await page.query_data(
                query,
                self.extract_data_timeout,
                self.wait_for_network_idle,
                self.include_hidden_data,
                self.response_mode,
            )
        elif prompt:
            return await page.get_data_by_prompt_experimental(
                prompt,
                self.extract_data_timeout,
                self.wait_for_network_idle,
                self.include_hidden_data,
                self.response_mode,
                request_origin="llamaindex-extractwebdata-browser-tool"
            )

    async def extract_web_element_with_browser(
        self,
        prompt: str,
    ) -> str:
        """
        Extract CSS selector of the target element the current web page on a running browser instance.
        The target element is identified by a natural language description.

        Args:
            prompt: The natural language description of the target element you want to extract.

        Returns:
            str: The CSS selector of the target element.
        """
        # Check if an async browser instance is provided    
        if not self.async_browser:
            raise ValueError("An async browser instance is required to use this tool.")
        
        page = await _aget_current_agentql_page(self.async_browser)
        element = await page.get_by_prompt(
            prompt,
            self.extract_elements_timeout,
            self.wait_for_network_idle,
            self.include_hidden_elements,
            self.response_mode,
            request_origin="llamaindex-extractwebelement-browser-tool"
        )
        tf_id = await element.get_attribute("tf623_id")
        return f"[tf623_id='{tf_id}']"