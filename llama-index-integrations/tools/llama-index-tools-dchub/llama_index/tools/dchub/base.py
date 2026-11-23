"""DC Hub tool spec: data center and energy intelligence for LlamaIndex agents."""

import json
import os
from typing import Any, Dict, List, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_ENDPOINT = "https://dchub.cloud/mcp"
USER_AGENT = "llama-index-tools-dchub/0.1.0"


class DCHubToolSpec(BaseToolSpec):
    """
    DC Hub tool spec.

    Calls the hosted DC Hub MCP server (Streamable HTTP, JSON-RPC ``tools/call``)
    and returns the server's text answer, which carries its as-of dates and a
    ``Source: DC Hub (dchub.cloud), as of <date>`` line for citation.

    A key is optional. Without one, some figures return at preview depth and the
    answer says so. Pass ``api_key`` or set ``DCHUB_API_KEY`` for fuller answers.
    """

    spec_functions = [
        "execute_plan",
        "get_market_dcpi_rank",
        "get_interconnection_queue",
        "get_grid_intelligence",
        "search_facilities",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 90.0,
    ) -> None:
        """Initialize with an optional API key and endpoint."""
        self.api_key = (
            api_key if api_key is not None else os.environ.get("DCHUB_API_KEY")
        )
        self.endpoint = endpoint
        self.timeout = timeout
        self._id = 0

    def _call(self, name: str, arguments: Dict[str, Any]) -> str:
        args = {k: v for k, v in arguments.items() if v is not None}
        self._id += 1
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        body = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        resp = requests.post(
            self.endpoint, json=body, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        payload = _parse_rpc(resp.text)
        if "error" in payload:
            err = payload["error"]
            return f"DC Hub error {err.get('code')}: {err.get('message')}"
        result = payload.get("result", {})
        texts: List[str] = [
            c.get("text", "")
            for c in result.get("content", [])
            if c.get("type") == "text"
        ]
        text = "\n".join(t for t in texts if t)
        if result.get("isError"):
            return f"DC Hub tool error: {text}"
        return text

    def execute_plan(
        self,
        intent: str,
        market: Optional[str] = None,
        iso: Optional[str] = None,
        state: Optional[str] = None,
        capacity_mw: Optional[float] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> str:
        """
        Answer a data center or energy question in one call. Use this first.

        Pass the user's question unchanged as ``intent``; DC Hub plans and runs the
        right lookups (power and grid, interconnection queues, Data Center Power
        Index (DCPI), fiber, facilities, site risk) and returns one dated answer.
        Keep each as-of date and end with the Source line the answer carries.

        Args:
            intent (str): The user's question, unchanged.
            market (Optional[str]): Market slug if known, e.g. "ashburn" or "columbus".
            iso (Optional[str]): ISO/RTO code if known, e.g. "PJM".
            state (Optional[str]): Two-letter US state if known, e.g. "VA".
            capacity_mw (Optional[float]): Target load in MW, e.g. 100.
            lat (Optional[float]): Latitude of a candidate site.
            lon (Optional[float]): Longitude of a candidate site.

        """
        return self._call(
            "execute_plan",
            {
                "intent": intent,
                "market": market,
                "iso": iso,
                "state": state,
                "capacity_mw": capacity_mw,
                "lat": lat,
                "lon": lon,
            },
        )

    def get_market_dcpi_rank(self, market_slug: str) -> str:
        """
        Get the Data Center Power Index (DCPI) verdict for one market.

        Args:
            market_slug (str): Market slug, e.g. "ashburn".

        """
        return self._call("get_market_dcpi_rank", {"market_slug": market_slug})

    def get_interconnection_queue(self, iso: Optional[str] = None) -> str:
        """
        Get interconnection queue depth and projects for an ISO/RTO.

        Args:
            iso (Optional[str]): ISO/RTO code, e.g. "PJM".

        """
        return self._call("get_interconnection_queue", {"iso": iso})

    def get_grid_intelligence(
        self, iso: Optional[str] = None, market: Optional[str] = None
    ) -> str:
        """
        Get current grid conditions (demand, supply mix, headroom) for an ISO or market.

        Args:
            iso (Optional[str]): ISO/RTO code, e.g. "PJM".
            market (Optional[str]): Market slug, e.g. "ashburn".

        """
        return self._call("get_grid_intelligence", {"iso": iso, "market": market})

    def search_facilities(
        self,
        query: Optional[str] = None,
        state: Optional[str] = None,
        city: Optional[str] = None,
        operator: Optional[str] = None,
        limit: Optional[int] = 10,
    ) -> str:
        """
        Search data center facilities by text, location or operator.

        Args:
            query (Optional[str]): Plain text over name, operator and location, e.g. "Ashburn".
            state (Optional[str]): Two-letter US state, e.g. "VA".
            city (Optional[str]): City name.
            operator (Optional[str]): Operator name.
            limit (Optional[int]): Maximum results (default 10).

        """
        return self._call(
            "search_facilities",
            {
                "query": query,
                "state": state,
                "city": city,
                "operator": operator,
                "limit": limit,
            },
        )


def _parse_rpc(text: str) -> Dict[str, Any]:
    """Parse a JSON-RPC response sent as plain JSON or as server-sent events."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    data_lines = [
        line[5:].strip() for line in text.splitlines() if line.startswith("data:")
    ]
    for chunk in reversed(data_lines):
        if chunk:
            try:
                parsed = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and ("result" in parsed or "error" in parsed):
                return parsed
    raise ValueError("DC Hub returned no JSON-RPC payload")
