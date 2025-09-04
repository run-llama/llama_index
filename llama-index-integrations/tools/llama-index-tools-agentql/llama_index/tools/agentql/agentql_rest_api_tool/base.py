from typing import Optional
import os

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.agentql.const import (
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_IS_STEALTH_MODE_ENABLED,
    DEFAULT_WAIT_FOR_PAGE_LOAD_SECONDS,
    DEFAULT_IS_SCROLL_TO_BOTTOM_ENABLED,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_IS_SCREENSHOT_ENABLED,
)
from llama_index.tools.agentql.messages import UNSET_API_KEY_ERROR_MESSAGE
from llama_index.tools.agentql.utils import _aload_data


class AgentQLRestAPIToolSpec(BaseToolSpec):
    """
    AgentQL Rest API Tool Spec.
    """

    spec_functions = [
        "extract_web_data_with_rest_api",
    ]

    def __init__(
        self,
        timeout: int = DEFAULT_API_TIMEOUT_SECONDS,
        is_stealth_mode_enabled: bool = DEFAULT_IS_STEALTH_MODE_ENABLED,
        wait_for: int = DEFAULT_WAIT_FOR_PAGE_LOAD_SECONDS,
        is_scroll_to_bottom_enabled: bool = DEFAULT_IS_SCROLL_TO_BOTTOM_ENABLED,
        mode: str = DEFAULT_RESPONSE_MODE,
        is_screenshot_enabled: bool = DEFAULT_IS_SCREENSHOT_ENABLED,
    ):
        """
        Initialize AgentQL Rest API Tool Spec.

        Args:
            timeout: The number of seconds to wait for a request before timing out. Defaults to 900.

            is_stealth_mode_enabled: Whether to enable experimental anti-bot evasion strategies. This feature may not work for all websites at all times.
            Data extraction may take longer to complete with this mode enabled. Defaults to `False`.

            wait_for: The number of seconds to wait for the page to load before extracting data. Defaults to 0.
            is_scroll_to_bottom_enabled: Whether to scroll to bottom of the page before extracting data. Defaults to `False`.

            mode: 'standard' uses deep data analysis, while 'fast' trades some depth of analysis for speed and is adequate for most usecases.
            Learn more about the modes in this guide: https://docs.agentql.com/accuracy/standard-mode) Defaults to 'fast'.

            is_screenshot_enabled: Whether to take a screenshot before extracting data. Returned in 'metadata' as a Base64 string. Defaults to `False`.

        """
        self._api_key = os.getenv("AGENTQL_API_KEY")
        if not self._api_key:
            raise ValueError(UNSET_API_KEY_ERROR_MESSAGE)
        self.timeout = timeout
        self.is_stealth_mode_enabled = is_stealth_mode_enabled
        self.wait_for = wait_for
        self.is_scroll_to_bottom_enabled = is_scroll_to_bottom_enabled
        self.mode = mode
        self.is_screenshot_enabled = is_screenshot_enabled

    async def extract_web_data_with_rest_api(
        self,
        url: str,
        query: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """
        Extracts structured data as a JSON from the active web page in a running browser instance using either an AgentQL query or a Natural Language description of the data.

        Args:
            url: URL of the public webpage to extract data from.
            query: AgentQL query used to extract the data. The query must be enclosed with curly braces `{}`. Either this field or `prompt` field must be provided.
            prompt: Natural Language description of the data to extract from the page. If AgentQL query is not specified, always use the `prompt` field. Either this field or `query` field must be provided.

        Returns:
            dict: Extracted data.

        """
        _params = {
            "wait_for": self.wait_for,
            "is_scroll_to_bottom_enabled": self.is_scroll_to_bottom_enabled,
            "mode": self.mode,
            "is_screenshot_enabled": self.is_screenshot_enabled,
        }
        _metadata = {
            "experimental_stealth_mode_enabled": self.is_stealth_mode_enabled,
        }

        return await _aload_data(
            url=url,
            query=query,
            prompt=prompt,
            params=_params,
            metadata=_metadata,
            api_key=self._api_key,
            timeout=self.timeout,
        )
