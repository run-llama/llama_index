from typing import Optional

from playwright.async_api import Browser as AsyncBrowser

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.agentql.const import (
    DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
    DEFAULT_WAIT_FOR_NETWORK_IDLE,
    DEFAULT_INCLUDE_HIDDEN_DATA,
    DEFAULT_RESPONSE_MODE,
    REQUEST_ORIGIN,
)
from llama_index.tools.agentql.messages import (
    QUERY_PROMPT_REQUIRED_ERROR_MESSAGE,
    QUERY_PROMPT_EXCLUSIVE_ERROR_MESSAGE,
)
from llama_index.tools.agentql.utils import _aget_current_agentql_page


class ExtractWebDataBrowserSpec(BaseToolSpec):
    """
    Extract web data from a browser.
    """

    spec_functions = [
        "extract_web_data_from_browser",
    ]

    def __init__(
        self,
        async_browser: AsyncBrowser,
        timeout: Optional[int] = DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
        wait_for_network_idle: Optional[bool] = DEFAULT_WAIT_FOR_NETWORK_IDLE,
        include_hidden: Optional[bool] = DEFAULT_INCLUDE_HIDDEN_DATA,
        mode: Optional[str] = DEFAULT_RESPONSE_MODE,
    ):
        """
        Initialize ExtractWebDataBrowserSpec.

        Args:
            async_browser: An async playwright browser instance.
            timeout: The number of seconds to wait for a request before timing out. Defaults to 900.
            wait_for_network_idle: Whether to wait until the network reaches a full idle state. Defaults to `True`.
            include_hidden: Whether to take into account visually hidden elements on the page. Defaults to `True`.

            mode: 'standard' uses deep data analysis, while 'fast' trades some depth of analysis for speed and is adequate for most usecases.
            Learn more about the modes in this guide: https://docs.agentql.com/accuracy/standard-mode. Defaults to 'fast'.
        """
        self.async_browser = async_browser
        self.timeout = timeout
        self.wait_for_network_idle = wait_for_network_idle
        self.include_hidden = include_hidden
        self.mode = mode

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
        # Check that query and prompt cannot be both empty or both provided
        if not query and not prompt:
            raise ValueError(QUERY_PROMPT_REQUIRED_ERROR_MESSAGE)
        if query and prompt:
            raise ValueError(QUERY_PROMPT_EXCLUSIVE_ERROR_MESSAGE)

        page = await _aget_current_agentql_page(self.async_browser)
        if query:
            return await page.query_data(
                query,
                self.timeout,
                self.wait_for_network_idle,
                self.include_hidden,
                self.mode,
                request_origin=REQUEST_ORIGIN,
            )
        else:
            return await page.get_data_by_prompt_experimental(
                prompt,
                self.timeout,
                self.wait_for_network_idle,
                self.include_hidden,
                self.mode,
                request_origin=REQUEST_ORIGIN,
            )
