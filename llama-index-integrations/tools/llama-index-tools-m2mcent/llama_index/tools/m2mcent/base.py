"""
M2MCent ToolSpec for LlamaIndex / LlamaHub
Enables autonomous AI agents in LlamaIndex to query, execute, and settle payments for 1,005 specialized tools via x402 on Base Mainnet.
"""

from typing import Any, Dict, List, Optional
import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

class M2MCentToolSpec(BaseToolSpec):
    """
    M2MCent Monolithic MCP & Agentic API Gateway ToolSpec.
    Provides 1,005 micro-services across finance, security, biotech, 3D, and IoT.
    Settled trustlessly via x402 V2 micropayments in USDC on Base Mainnet.
    """

    spec_functions = [
        "search_tools",
        "get_tool_info",
        "execute_tool"
    ]

    def __init__(self, base_url: str = "https://api.m2mcent.com", auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token

    def search_tools(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search available M2MCent micro-services by keyword or semantic domain.
        
        Args:
            query: The domain, keyword, or action to search for (e.g., 'credit-risk', 'uniswap', 'solidity').
            limit: Maximum number of tools to return.
        """
        resp = requests.get(f"{self.base_url}/api/v1/search", params={"q": query, "limit": limit})
        if resp.status_code == 200:
            return resp.json().get("results", [])
        return []

    def get_tool_info(self, tool_id: str) -> Dict[str, Any]:
        """
        Retrieve schema, price, and payment challenge for a specific M2MCent tool.
        
        Args:
            tool_id: Unique slug of the tool (e.g., 'credit-risk-mcp').
        """
        resp = requests.get(f"{self.base_url}/{tool_id}/server-card.json")
        if resp.status_code == 200:
            return resp.json()
        return {"error": "Tool not found", "status": resp.status_code}

    def execute_tool(self, tool_id: str, payload: Dict[str, Any], payment_signature: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute an M2MCent tool with payload and optional x402 payment signature.
        
        Args:
            tool_id: Unique slug of the tool.
            payload: JSON input payload matching the tool schema.
            payment_signature: Base64-encoded EIP-3009 authorization signature on Base Mainnet.
        """
        headers = {"Content-Type": "application/json"}
        if payment_signature:
            headers["Payment-Signature"] = payment_signature
        elif self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        resp = requests.post(f"{self.base_url}/{tool_id}/api/process", json=payload, headers=headers)
        return resp.json()
