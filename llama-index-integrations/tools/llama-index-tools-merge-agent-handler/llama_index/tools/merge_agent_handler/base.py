"""Merge Agent Handler tool spec."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from llama_index.core.tools.tool_spec.base import BaseToolSpec

logger = logging.getLogger(__name__)

BASE_URL = "https://ah-api.merge.dev/api/v1"


class MergeAgentHandlerToolSpec(BaseToolSpec):
    """
    Merge Agent Handler tool spec.

    Connects AI agents to Merge Agent Handler Tool Packs via the Model
    Context Protocol (MCP). Provides access to pre-built integrations
    across HRIS, ATS, CRM, accounting, ticketing, and file storage.
    """

    spec_functions = [
        "list_tool_packs",
        "list_registered_users",
        "list_tools",
        "call_tool",
    ]

    def __init__(
        self,
        api_key: str,
        tool_pack_id: Optional[str] = None,
        registered_user_id: Optional[str] = None,
        environment: str = "production",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize with Merge Agent Handler credentials.

        Args:
            api_key: Merge Agent Handler API key.
            tool_pack_id: Default Tool Pack ID (can be overridden per-call).
            registered_user_id: Default Registered User ID (can be overridden per-call).
            environment: Either "production" or "test". Defaults to "production".
            timeout: HTTP timeout in seconds.

        """
        self.api_key = api_key
        self.tool_pack_id = tool_pack_id
        self.registered_user_id = registered_user_id
        self.environment = environment
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Source": "llamaindex-tool",
        }
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers=self._headers,
        )

    def close(self) -> None:
        """Close the shared HTTP client."""
        self._client.close()

    def _resolve_environment(self, environment: Optional[str]) -> str:
        resolved = (environment or self.environment or "production").strip().lower()
        if resolved not in {"production", "test"}:
            raise ValueError("environment must be either 'production' or 'test'")
        return resolved

    def _resolve_identifiers(
        self,
        tool_pack_id: Optional[str],
        registered_user_id: Optional[str],
    ) -> tuple[str, str]:
        resolved_tool_pack_id = tool_pack_id or self.tool_pack_id
        resolved_registered_user_id = registered_user_id or self.registered_user_id

        if not resolved_tool_pack_id:
            raise ValueError("tool_pack_id is required. Pass it in __init__ or this method call.")
        if not resolved_registered_user_id:
            raise ValueError("registered_user_id is required. Pass it in __init__ or this method call.")

        return resolved_tool_pack_id, resolved_registered_user_id

    def _parse_arguments(self, arguments: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("arguments must be a valid JSON string representing an object") from exc

        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ValueError("arguments must decode to a JSON object")

        return parsed

    def _fetch_all_pages(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        all_results: List[Dict[str, Any]] = []
        page = 1
        base_params = dict(params or {})

        while True:
            response = self._client.get(url, params={**base_params, "page": str(page)})
            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]

            if not isinstance(payload, dict):
                raise ValueError("Unexpected API response format")

            results = payload.get("results", [])
            if not isinstance(results, list):
                raise ValueError("Unexpected paginated response format")

            all_results.extend(item for item in results if isinstance(item, dict))

            if not payload.get("next"):
                break
            page += 1

        return all_results

    def _post_mcp(
        self,
        tool_pack_id: str,
        registered_user_id: str,
        rpc_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = self._client.post(
            f"/tool-packs/{tool_pack_id}/registered-users/{registered_user_id}/mcp",
            json=rpc_request,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected MCP response format")
        return payload

    @staticmethod
    def _extract_text(content: Any) -> str:
        if not isinstance(content, list):
            return ""

        text_chunks: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_chunks.append(item["text"])
        return "\n".join(text_chunks).strip()

    def list_tool_packs(self) -> str:
        """
        List all available Merge Agent Handler Tool Packs.

        Fetches the Tool Packs configured in your Merge Agent Handler account.
        Each Tool Pack bundles connectors for services like Jira, Salesforce,
        Greenhouse, etc. Use the returned tool pack IDs with list_tools and call_tool.

        Returns:
            str: JSON array of tool packs, each with "id", "name", "description",
                and "connectors" (list of connected services).

        """
        tool_packs = self._fetch_all_pages("/tool-packs/")

        normalized_tool_packs: List[Dict[str, Any]] = []
        for tool_pack in tool_packs:
            raw_connectors = tool_pack.get("connectors")
            connectors: List[Dict[str, Any]] = raw_connectors if isinstance(raw_connectors, list) else []
            normalized_tool_packs.append(
                {
                    "id": tool_pack.get("id"),
                    "name": tool_pack.get("name"),
                    "description": tool_pack.get("description"),
                    "connectors": [
                        {"name": connector.get("name"), "slug": connector.get("slug")}
                        for connector in connectors
                        if isinstance(connector, dict)
                    ],
                }
            )

        return json.dumps(normalized_tool_packs)

    def list_registered_users(self, environment: Optional[str] = None) -> str:
        """
        List registered users for Merge Agent Handler.

        Args:
            environment: "production" or "test". Defaults to the value set in constructor.

        Returns:
            str: JSON array of registered users with "id", "origin_user_name",
                and "authenticated_connectors".

        """
        resolved_environment = self._resolve_environment(environment)
        is_test = resolved_environment == "test"
        users = self._fetch_all_pages("/registered-users", {"is_test": str(is_test).lower()})

        normalized_users = [
            {
                "id": user.get("id"),
                "origin_user_name": user.get("origin_user_name"),
                "authenticated_connectors": user.get("authenticated_connectors"),
            }
            for user in users
        ]

        return json.dumps(normalized_users)

    def list_tools(
        self,
        tool_pack_id: Optional[str] = None,
        registered_user_id: Optional[str] = None,
    ) -> str:
        """
        List available MCP tools in a Merge Tool Pack.

        Args:
            tool_pack_id: Tool Pack ID. Uses default from constructor if not provided.
            registered_user_id: Registered User ID. Uses default from constructor if not provided.

        Returns:
            str: JSON array of tools, each with "name" and "description".

        """
        resolved_tool_pack_id, resolved_registered_user_id = self._resolve_identifiers(
            tool_pack_id,
            registered_user_id,
        )

        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }

        response = self._post_mcp(resolved_tool_pack_id, resolved_registered_user_id, rpc_request)

        if "error" in response:
            error = response.get("error", {})
            if isinstance(error, dict):
                raise ValueError(f"MCP tools/list failed: {error.get('message', 'Unknown error')}")
            raise ValueError("MCP tools/list failed: Unknown error")

        result = response.get("result", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        if not isinstance(tools, list):
            raise ValueError("Unexpected tools payload format")

        normalized_tools = [
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
            }
            for tool in tools
            if isinstance(tool, dict)
        ]
        return json.dumps(normalized_tools)

    def call_tool(
        self,
        tool_name: str,
        arguments: str = "{}",
        tool_pack_id: Optional[str] = None,
        registered_user_id: Optional[str] = None,
    ) -> str:
        """
        Execute an MCP tool from a Merge Tool Pack.

        Args:
            tool_name: The name of the MCP tool to execute (from list_tools).
            arguments: JSON string of arguments to pass to the tool.
            tool_pack_id: Tool Pack ID. Uses default from constructor if not provided.
            registered_user_id: Registered User ID. Uses default from constructor if not provided.

        Returns:
            str: The tool's text output, or an error message if the call failed.

        """
        try:
            resolved_tool_pack_id, resolved_registered_user_id = self._resolve_identifiers(
                tool_pack_id,
                registered_user_id,
            )
            parsed_arguments = self._parse_arguments(arguments)

            rpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": {"input": parsed_arguments},
                },
            }

            response = self._post_mcp(resolved_tool_pack_id, resolved_registered_user_id, rpc_request)
        except Exception as exc:
            logger.exception(exc)
            return f'Error calling tool "{tool_name}": {exc}'

        if "error" in response:
            error = response.get("error", {})
            if isinstance(error, dict):
                return f'Tool "{tool_name}" returned error: {error.get("message", "Unknown error")}'
            return f'Tool "{tool_name}" returned error: Unknown error'

        result = response.get("result")
        if not isinstance(result, dict):
            return json.dumps(result)

        if result.get("isError"):
            error_text = self._extract_text(result.get("content")) or "Unknown error"
            return f'Tool "{tool_name}" failed: {error_text}'

        content = result.get("content")
        text_output = self._extract_text(content)
        if text_output:
            return text_output

        return json.dumps(result)
