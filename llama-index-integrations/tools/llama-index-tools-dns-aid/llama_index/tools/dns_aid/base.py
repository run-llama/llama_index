"""DNS-AID tool specification for LlamaIndex.

Provides discover, publish, and unpublish operations as LlamaIndex FunctionTool objects.
"""

from __future__ import annotations

from typing import Any, Optional


class DnsAidToolSpec:
    """DNS-AID tool spec for LlamaIndex.

    Example::

        from llama_index_tools_dns_aid import DnsAidToolSpec

        spec = DnsAidToolSpec(backend_name="route53")
        tools = spec.to_tool_list()
        # Use tools in a LlamaIndex agent
    """

    def __init__(
        self,
        backend_name: Optional[str] = None,
        backend: Any = None,
    ) -> None:
        from dns_aid.integrations import DnsAidOperations

        self._ops = DnsAidOperations(backend_name=backend_name, backend=backend)

    def discover_agents(
        self,
        domain: str,
        protocol: Optional[str] = None,
        name: Optional[str] = None,
        require_dnssec: bool = False,
    ) -> str:
        """Discover AI agents at a domain via DNS-AID SVCB records."""
        return self._ops.discover_sync(
            domain=domain, protocol=protocol, name=name, require_dnssec=require_dnssec
        )

    def publish_agent(
        self,
        name: str,
        domain: str,
        protocol: str = "mcp",
        endpoint: str = "",
        port: int = 443,
        capabilities: Optional[list[str]] = None,
        version: str = "1.0.0",
        description: Optional[str] = None,
        ttl: int = 3600,
    ) -> str:
        """Publish an AI agent to DNS via DNS-AID."""
        return self._ops.publish_sync(
            name=name,
            domain=domain,
            protocol=protocol,
            endpoint=endpoint,
            port=port,
            capabilities=capabilities,
            version=version,
            description=description,
            ttl=ttl,
        )

    def unpublish_agent(
        self,
        name: str,
        domain: str,
        protocol: str = "mcp",
    ) -> str:
        """Remove an AI agent from DNS via DNS-AID."""
        return self._ops.unpublish_sync(name=name, domain=domain, protocol=protocol)

    def to_tool_list(self) -> list:
        """Convert to list of LlamaIndex FunctionTool objects."""
        from llama_index.core.tools import FunctionTool

        return [
            FunctionTool.from_defaults(
                fn=self.discover_agents,
                name="dns_aid_discover",
                description="Discover AI agents at a domain via DNS-AID SVCB records.",
            ),
            FunctionTool.from_defaults(
                fn=self.publish_agent,
                name="dns_aid_publish",
                description="Publish an AI agent to DNS via DNS-AID.",
            ),
            FunctionTool.from_defaults(
                fn=self.unpublish_agent,
                name="dns_aid_unpublish",
                description="Remove an AI agent from DNS via DNS-AID.",
            ),
        ]
