from typing import Optional, List
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from hive_intelligence.client import HiveSearchClient
from hive_intelligence.types import HiveSearchRequest, HiveSearchMessage, HiveSearchResponse
from hive_intelligence.errors import HiveSearchAPIError

class HiveToolSpec(BaseToolSpec):
    """Hive Search tool spec."""

    spec_functions = ["search"]

    def __init__(self, api_key: str) -> None:
        self.client = HiveSearchClient(api_key=api_key)

    def search(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[HiveSearchMessage]] = None,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        include_data_sources: bool = False
    ) -> HiveSearchResponse:
        """
        Executes a Hive search request via prompt or chat-style messages.
        """
        req = HiveSearchRequest(
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            include_data_sources=include_data_sources,
        )
        try:
            response= self.client.search(req)
        except HiveSearchAPIError as e:
            raise RuntimeError(f"{e}") from e

        # Return the Hive search response
        return response
