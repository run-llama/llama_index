"""
Basic A2A Client for connecting to A2A agents.

Mirrors the BasicMCPClient pattern but adapted for the A2A protocol.
Handles AgentCard discovery and task-based skill invocation.
"""

import warnings
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Callable, AsyncIterator, Awaitable, Any
from urllib.parse import urlparse

import httpx
from pydantic import AnyUrl

from llama_index.tools.a2a.models import AgentCard, TaskResult, AgentCapabilities, AgentProvider


logger = logging.getLogger(__name__)


class DefaultInMemoryTokenStorage:
    """Simple in-memory token storage for A2A authentication."""

    def __init__(self):
        self._tokens: Optional[dict] = None
        self._client_info: Optional[dict] = None

    async def get_tokens(self) -> Optional[dict]:
        return self._tokens

    async def set_tokens(self, tokens: dict) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> Optional[dict]:
        return self._client_info

    async def set_client_info(self, client_info: dict) -> None:
        self._client_info = client_info


class BasicA2AClient:
    """
    Basic A2A client for connecting to an A2A agent.

    This client discovers the agent's capabilities via its AgentCard
    and provides methods to invoke skills through the A2A protocol.

    Args:
        agent_card_url: The URL of the agent card (e.g., https://agent.example.com/.well-known/agent-card.json).
        timeout: The timeout for HTTP operations in seconds. Default is 30.
        headers: Optional headers to pass with requests.
        auth: Optional authentication handler.
        tool_call_logs_callback: Async function to capture logs from skill calls.

    """

    def __init__(
        self,
        agent_card_url: str,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Any] = None,
        tool_call_logs_callback: Optional[Callable[[List[str]], Awaitable[Any]]] = None,
    ):
        self.agent_card_url = agent_card_url
        self.timeout = timeout
        self.headers = headers or {}
        self.auth = auth
        self.tool_call_logs_callback = tool_call_logs_callback

        # Cached agent card
        self._agent_card: Optional[AgentCard] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create an HTTP client with configured timeouts."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self.headers,
            )
        return self._http_client

    async def _discover_agent_card(self) -> AgentCard:
        """Discover and parse the agent card from the given URL."""
        if self._agent_card is not None:
            return self._agent_card

        client = await self._get_http_client()
        response = await client.get(self.agent_card_url)
        response.raise_for_status()

        self._agent_card = AgentCard.model_validate(response.json())
        return self._agent_card

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @asynccontextmanager
    async def _session(self) -> AsyncIterator["BasicA2AClient"]:
        """Context manager for session lifecycle."""
        try:
            yield self
        finally:
            await self.close()

    # ---- Discovery Methods ----

    async def get_agent_card(self) -> AgentCard:
        """
        Get the agent's capabilities from its AgentCard.

        Returns:
            The parsed AgentCard containing agent metadata and skills.
        """
        return await self._discover_agent_card()

    async def list_skills(self) -> List[Any]:
        """
        List all available skills on the A2A agent.

        Returns:
            A list of skill descriptors with name, description, and input_schema.
        """
        card = await self._discover_agent_card()
        return card.skills

    # ---- Skill Invocation Methods ----

    async def call_skill(
        self,
        skill_name: str,
        arguments: Optional[dict] = None,
    ) -> TaskResult:
        """
        Call a specific skill on the A2A agent.

        This method creates a task, sends it to the agent, and polls
        for the result using the A2A task protocol.

        Args:
            skill_name: The name of the skill to invoke.
            arguments: Optional arguments to pass to the skill.

        Returns:
            The task result containing the output.
        """
        card = await self._discover_agent_card()

        # Find the skill endpoint
        skill_endpoint = None
        for skill in card.skills:
            if skill.name == skill_name:
                skill_endpoint = skill.endpoint or card.url
                break

        if skill_endpoint is None:
            skill_endpoint = card.url

        client = await self._get_http_client()

        # Build the A2A request payload
        request_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "task": {
                    "name": skill_name,
                    "input": {
                        "parts": [
                            {
                                "kind": "text",
                                "text": str(arguments) if arguments else ""
                            }
                        ]
                    }
                }
            }
        }

        # Send the task
        response = await client.post(
            skill_endpoint,
            json=request_payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        # Parse the response
        result = response.json()
        raw_result = result.get("result", {})
        # Map 'id' from JSON-RPC response to 'task_id' for TaskResult
        if "id" in raw_result and "task_id" not in raw_result:
            raw_result["task_id"] = raw_result.pop("id")
        return TaskResult.model_validate(raw_result)

    async def send_message(
        self,
        message: str,
        context_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Send a natural language message to the agent.

        Args:
            message: The message to send.
            context_id: Optional context ID for stateful conversations.

        Returns:
            The agent's response as a TaskResult.
        """
        card = await self._discover_agent_card()

        client = await self._get_http_client()

        request_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "messages/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                }
            }
        }

        if context_id:
            request_payload["params"]["contextId"] = context_id

        response = await client.post(
            card.url,
            json=request_payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        result = response.json()
        raw_result = result.get("result", {})
        # Map 'id' from JSON-RPC response to 'task_id' for TaskResult
        if "id" in raw_result and "task_id" not in raw_result:
            raw_result["task_id"] = raw_result.pop("id")
        return TaskResult.model_validate(raw_result)
