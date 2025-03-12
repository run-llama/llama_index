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
        timeout: Optional[int] = DEFAULT_API_TIMEOUT_SECONDS,
    ):
        """
        Initialize AgentQL Rest API Tool Spec.

        Args:
            timeout: The number of seconds to wait for a request before timing out. **Defaults to `900`.**
        """
        self._api_key = os.getenv("AGENTQL_API_KEY")
        if not self._api_key:
            raise ValueError(UNSET_API_KEY_ERROR_MESSAGE)
        self.timeout = timeout

    async def extract_web_data_with_rest_api(
        self,
        url: str,
        query: Optional[str] = None,
        prompt: Optional[str] = None,
        is_stealth_mode_enabled: Optional[bool] = DEFAULT_IS_STEALTH_MODE_ENABLED,
        wait_for: Optional[int] = DEFAULT_WAIT_FOR_PAGE_LOAD_SECONDS,
        is_scroll_to_bottom_enabled: Optional[
            bool
        ] = DEFAULT_IS_SCROLL_TO_BOTTOM_ENABLED,
        mode: Optional[str] = DEFAULT_RESPONSE_MODE,
        is_screenshot_enabled: Optional[bool] = DEFAULT_IS_SCREENSHOT_ENABLED,
    ) -> dict:
        """
        Extracts structured JSON data from a webpage URL using an AgentQL query or Natural Language Description.

        Args:
            url: Webpage URL to extract data from.
            query: AgentQL query enclosed in `{}`. One of `query` or `prompt` required.
            prompt: Natural Language description of data to extract.

            timeout: Seconds before request times out.
            is_stealth_mode_enabled: enable experimental anti-bot evasion strategies.
            wait_for: Seconds to wait for page load.
            is_scroll_to_bottom_enabled: Scroll to bottom before extraction.
            mode: 'standard' provides deeper analysis, 'fast' prioritizes speed.
            is_screenshot_enabled: take screenshot, return as Base64.

        Returns:
            dict: Extracted data.
        """
        _params = {
            "wait_for": wait_for,
            "is_scroll_to_bottom_enabled": is_scroll_to_bottom_enabled,
            "mode": mode,
            "is_screenshot_enabled": is_screenshot_enabled,
        }
        _metadata = {
            "experimental_stealth_mode_enabled": is_stealth_mode_enabled,
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
