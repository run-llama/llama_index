from typing import Optional, List
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from hive_intelligence.client import HiveSearchClient
from hive_intelligence.types import (
    HiveSearchRequest,
    HiveSearchMessage,
    HiveSearchResponse,
)
from hive_intelligence.errors import HiveSearchAPIError


class HiveToolSpec(BaseToolSpec):
    """Hive Search tool spec."""

    spec_functions = ["search"]

    def __init__(
        self,
        api_key: str,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> None:
        self.client = HiveSearchClient(api_key=api_key)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def search(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[HiveSearchMessage]] = None,
        include_data_sources: bool = False,
    ) -> HiveSearchResponse:
        """
        Executes a Hive search request via prompt or chat-style messages.
        """
        req_args = {
            "prompt": prompt,
            "messages": messages,
            "include_data_sources": include_data_sources,
        }

        # Only add parameters if they are not None
        if self.temperature is not None:
            req_args["temperature"] = self.temperature
        if self.top_k is not None:
            req_args["top_k"] = self.top_k
        if self.top_p is not None:
            req_args["top_p"] = self.top_p

        req = HiveSearchRequest(**req_args)
        try:
            response = self.client.search(req)
        except HiveSearchAPIError as e:
            raise RuntimeError(f"{e}") from e

        # Return the Hive search response
        return response
