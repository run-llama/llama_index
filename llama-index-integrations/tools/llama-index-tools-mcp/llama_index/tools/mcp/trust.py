"""Optional MCP server trust verification before tool execution.

This module provides a lightweight, optional pre-call trust gate that
agents can use to check an MCP server's trust score before connecting.

The trust check is pluggable: pass any callable that takes a server URL
and returns a trust verdict (0-1 float or None for unknown). The default
implementation queries a public trust API, but users can substitute their
own (local, cached, or air-gapped) verifier.

Usage:

    from llama_index.tools.mcp.trust import TrustChecker, TrustVerdict

    # Optional: trust-check before connecting
    checker = TrustChecker(min_score=0.5)
    verdict = await checker.check("https://example.com/mcp/sse")
    if not verdict.is_trusted:
        raise ValueError(f"Server not trusted: {verdict.reason}")

    # Or use the convenience wrapper that skips untrusted servers silently
    trusted = await checker.filter_trusted(["https://a.com/sse", "https://b.com/sse"])
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Awaitable

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TrustVerdict:
    """The result of a trust check on one MCP server."""

    server_url: str
    score: Optional[float]  # 0.0 to 1.0, or None if unknown
    total_calls: Optional[int] = None
    is_trusted: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.score is not None and self.score >= 0 and self.score <= 1:
            pass  # score is valid
        elif self.score is not None:
            self.reason = f"Invalid score format: {self.score}"
            self.score = None


# Default trust API endpoint (free tier, 50 queries/day, no key needed)
_DEFAULT_TRUST_API = "https://dominionobservatory.com/api/trust"


async def _default_trust_lookup(
    server_url: str,
    api_url: str = _DEFAULT_TRUST_API,
    timeout: float = 5.0,
) -> TrustVerdict:
    """Query the default public trust API for a server's trust score.

    This is the default implementation — users can pass their own
    callable to TrustChecker for local or air-gapped verification.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(api_url, params={"url": server_url})
            if resp.status_code != 200:
                return TrustVerdict(
                    server_url=server_url,
                    score=None,
                    reason=f"Trust API returned {resp.status_code}",
                )
            data = resp.json()
            score = data.get("trust_score")
            total_calls = data.get("total_calls", 0)
            # Normalize: some APIs return 0-100, some return 0-1
            if score is not None and score > 1:
                score = score / 100.0
            return TrustVerdict(
                server_url=server_url,
                score=score,
                total_calls=total_calls,
            )
    except (httpx.RequestError, asyncio.TimeoutError) as e:
        logger.debug("Trust check failed for %s: %s", server_url, e)
        return TrustVerdict(
            server_url=server_url,
            score=None,
            reason=f"Trust API unreachable: {e}",
        )


# Type for a pluggable trust lookup function
TrustLookup = Callable[[str], Awaitable[TrustVerdict]]


class TrustChecker:
    """Optional pre-call trust gate for MCP server connections.

    Usage:

        checker = TrustChecker(min_score=0.5)

        # Check one server
        verdict = await checker.check("https://example.com/mcp/sse")
        if not verdict.is_trusted:
            print(f"Skipping untrusted server: {verdict.reason}")

        # Filter a list of server URLs, keeping only trusted ones
        trusted_urls = await checker.filter_trusted([
            "https://a.com/sse",
            "https://b.com/sse",
        ])

    Args:
        min_score: Minimum trust score to consider a server trusted (default 0.5).
        lookup: Custom trust lookup function. Defaults to the public Dominion
                 Observatory API (free tier, 50 queries/day).
        allow_unknown: If True, servers with unknown trust scores are considered
                       trusted (fail-open). If False (default), unknown servers
                       are considered untrusted (fail-closed).
    """

    def __init__(
        self,
        min_score: float = 0.5,
        lookup: Optional[TrustLookup] = None,
        allow_unknown: bool = False,
    ) -> None:
        self.min_score = min_score
        self._lookup = lookup or _default_trust_lookup
        self.allow_unknown = allow_unknown

    async def check(self, server_url: str) -> TrustVerdict:
        """Check the trust score of a single MCP server.

        Returns a TrustVerdict with the score and whether it's trusted.
        """
        verdict = await self._lookup(server_url)
        if verdict.score is None:
            verdict.is_trusted = self.allow_unknown
            if not verdict.is_trusted and not verdict.reason:
                verdict.reason = "Trust score unknown (fail-closed)"
        else:
            verdict.is_trusted = verdict.score >= self.min_score
            if not verdict.is_trusted:
                verdict.reason = (
                    f"Score {verdict.score:.2f} below threshold {self.min_score:.2f}"
                )
        return verdict

    async def filter_trusted(
        self, server_urls: List[str]
    ) -> List[str]:
        """Filter a list of server URLs, keeping only the trusted ones.

        Servers with unknown trust scores are kept or dropped depending on
        the `allow_unknown` setting.
        """
        results = await asyncio.gather(
            *[self.check(url) for url in server_urls],
            return_exceptions=True,
        )
        trusted: List[str] = []
        for url, result in zip(server_urls, results):
            if isinstance(result, Exception):
                logger.warning("Trust check raised for %s: %s", url, result)
                if self.allow_unknown:
                    trusted.append(url)
                continue
            if result.is_trusted:
                trusted.append(url)
            else:
                logger.info("Skipping untrusted MCP server %s: %s", url, result.reason)
        return trusted
