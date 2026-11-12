"""
A2A (Agent-to-Agent) Protocol integration for LlamaIndex.

This module provides outbound and inbound A2A protocol support,
mirroring the MCP integration pattern.

Outbound: Connect to an A2A agent card URL and discover tools
Inbound: Expose a Workflow as an A2A-capable ASGI app
"""

from llama_index.tools.a2a.base import A2AToolSpec
from llama_index.tools.a2a.client import BasicA2AClient
from llama_index.tools.a2a.server import (
    workflow_as_a2a,
    create_agent_card,
    WorkflowAgentExecutor,
)
from llama_index.tools.a2a.utils import (
    get_tools_from_a2a_card,
    aget_tools_from_a2a_card,
)

__all__ = [
    "A2AToolSpec",
    "BasicA2AClient",
    "workflow_as_a2a",
    "create_agent_card",
    "WorkflowAgentExecutor",
    "get_tools_from_a2a_card",
    "aget_tools_from_a2a_card",
]
