from typing import Optional

from playwright.async_api import Browser as AsyncBrowser

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.agentql.const import (
    DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
    DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
    DEFAULT_WAIT_FOR_NETWORK_IDLE,
    DEFAULT_INCLUDE_HIDDEN_DATA,
    DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
    DEFAULT_RESPONSE_MODE,
    REQUEST_ORIGIN,
)
from llama_index.tools.agentql.messages import (
    QUERY_PROMPT_REQUIRED_ERROR_MESSAGE,
    QUERY_PROMPT_EXCLUSIVE_ERROR_MESSAGE,
)
from llama_index.tools.agentql.utils import _aget_current_agentql_page


class AgentQLBrowserToolSpec(BaseToolSpec):
    """
    AgentQL Browser Tool Spec.
    """

    spec_functions = [
        "extract_web_data_from_browser",
        "get_web_element_from_browser",
    ]

    def __init__(
        self,
        async_browser: AsyncBrowser,
        extract_data_timeout: Optional[int] = DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
        get_element_timeout: Optional[int] = DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
    ):
        """
        Initialize ExtractWebDataBrowserSpec.

        Args:
            async_browser: An async playwright browser instance.
            extract_data_timeout: The number of seconds to wait for a request before data extraction times out. **Defaults to `900`.**
            get_element_timeout: The number of seconds to wait for a request before getting element times out. **Defaults to `300`.**
        """
        self.async_browser = async_browser
        self.extract_data_timeout = extract_data_timeout
        self.get_element_timeout = get_element_timeout

    async def extract_web_data_from_browser(
        self,
        query: Optional[str] = None,
        prompt: Optional[str] = None,
        wait_for_network_idle: Optional[bool] = DEFAULT_WAIT_FOR_NETWORK_IDLE,
        include_hidden: Optional[bool] = DEFAULT_INCLUDE_HIDDEN_DATA,
        mode: Optional[str] = DEFAULT_RESPONSE_MODE,
    ) -> dict:
        """
        Extracts structured data as JSON from the active web page in a running browser instance using an AgentQL query or Natural Language description.

        Args:
            query: AgentQL query enclosed in `{}`. One of `query` or `prompt` required.
            prompt: Natural Language description of the data to extract.

            wait_for_network_idle: Whether to wait for network idle state.
            include_hidden: Whether to include visually hidden elements.
            mode: 'standard' uses deep data analysis, 'fast' trades analysis depth for speed.

        Returns:
            dict: Extracted data
        """
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
                wait_for_network_idle,
                include_hidden,
                mode,
                request_origin=REQUEST_ORIGIN,
            )
        else:
            return await page.get_data_by_prompt_experimental(
                prompt,
                self.extract_data_timeout,
                wait_for_network_idle,
                include_hidden,
                mode,
                request_origin=REQUEST_ORIGIN,
            )

    async def get_web_element_from_browser(
        self,
        prompt: str,
        wait_for_network_idle: Optional[bool] = DEFAULT_WAIT_FOR_NETWORK_IDLE,
        include_hidden: Optional[bool] = DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
        mode: Optional[str] = DEFAULT_RESPONSE_MODE,
    ) -> str:
        """
        Finds a web element on the active web page in a running browser instance using element’s Natural Language description and returns its CSS selector for further interaction, like clicking, filling a form field, etc.

        Args:
            prompt: Natural Language description of the web element

            wait_for_network_idle: Whether to wait for network idle state.
            include_hidden: Whether to include visually hidden elements.
            mode: 'standard' uses deep data analysis, 'fast' trades analysis depth for speed.

        Returns:
            str: The CSS selector of the target element.
        """
        page = await _aget_current_agentql_page(self.async_browser)
        element = await page.get_by_prompt(
            prompt,
            self.get_element_timeout,
            wait_for_network_idle,
            include_hidden,
            mode,
            request_origin=REQUEST_ORIGIN,
        )
        tf_id = await element.get_attribute("tf623_id")
        return f"[tf623_id='{tf_id}']"
