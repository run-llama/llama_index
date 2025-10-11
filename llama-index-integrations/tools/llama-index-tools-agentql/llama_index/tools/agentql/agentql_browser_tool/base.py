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
        timeout_for_data: int = DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
        timeout_for_element: int = DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
        wait_for_network_idle: bool = DEFAULT_WAIT_FOR_NETWORK_IDLE,
        include_hidden_for_data: bool = DEFAULT_INCLUDE_HIDDEN_DATA,
        include_hidden_for_element: bool = DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
        mode: str = DEFAULT_RESPONSE_MODE,
    ):
        """
        Initialize AgentQL Browser Tool Spec.

        Args:
            async_browser: An async playwright browser instance.
            timeout_for_data: The number of seconds to wait for a extract data request before timing out. Defaults to 900.
            timeout_for_element: The number of seconds to wait for a get element request before timing out. Defaults to 300.
            wait_for_network_idle: Whether to wait for network idle state. Defaults to `True`.
            include_hidden_for_data: Whether to take into account visually hidden elements on the page for extract data. Defaults to `True`.
            include_hidden_for_element: Whether to take into account visually hidden elements on the page for get element. Defaults to `False`.

            mode: `standard` uses deep data analysis, while `fast` trades some depth of analysis for speed and is adequate for most usecases.
            Learn more about the modes in this guide: https://docs.agentql.com/accuracy/standard-mode. Defaults to `fast`.

        """
        self.async_browser = async_browser
        self.timeout_for_data = timeout_for_data
        self.timeout_for_element = timeout_for_element
        self.wait_for_network_idle = wait_for_network_idle
        self.include_hidden_for_data = include_hidden_for_data
        self.include_hidden_for_element = include_hidden_for_element
        self.mode = mode

    async def extract_web_data_from_browser(
        self,
        query: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """
        Extracts structured data as JSON from a web page given a URL using either an AgentQL query or a Natural Language description of the data.

        Args:
            query: AgentQL query used to extract the data. The query must be enclosed with curly braces `{}`. Either this field or `prompt` field must be provided.
            prompt: Natural Language description of the data to extract from the page. If AgentQL query is not specified, always use the `prompt` field. Either this field or `query` field must be provided.

        Returns:
            dict: The extracted data

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
                self.timeout_for_data,
                self.wait_for_network_idle,
                self.include_hidden_for_data,
                self.mode,
                request_origin=REQUEST_ORIGIN,
            )
        else:
            return await page.get_data_by_prompt_experimental(
                prompt,
                self.timeout_for_data,
                self.wait_for_network_idle,
                self.include_hidden_for_data,
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
            prompt: Natural Language description of the web element to find on the page.

        Returns:
            str: The CSS selector of the target element.

        """
        page = await _aget_current_agentql_page(self.async_browser)
        element = await page.get_by_prompt(
            prompt,
            self.timeout_for_element,
            self.wait_for_network_idle,
            self.include_hidden_for_element,
            self.mode,
            request_origin=REQUEST_ORIGIN,
        )
        tf_id = await element.get_attribute("tf623_id")
        return f"[tf623_id='{tf_id}']"
