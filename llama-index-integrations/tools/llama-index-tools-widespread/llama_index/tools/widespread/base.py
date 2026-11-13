"""Widespread protocol tool spec — discovery and status reads for verified-work
bounty markets. The write/earn loop (signed Ed25519 attestations) lives behind
the Widespread MCP server; this spec intentionally exposes only safe reads."""

import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE = "https://widespread-bridge.widespread.workers.dev"


class WidespreadToolSpec(BaseToolSpec):
    """Widespread verified-work bounties: funded-market discovery, feed search,
    sealed-runner directory, claim/payout status. Non-custodial — no signing
    material is accepted or held by this spec."""

    spec_functions = [
        "list_bounties",
        "list_markets",
        "feed_search",
        "get_feed_item",
        "list_runners",
        "get_state",
    ]

    def __init__(self, base_url: str = DEFAULT_BASE, timeout_seconds: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _get(self, path: str) -> Any:
        req = Request(
            f"{self.base_url}{path}", headers={"accept": "application/json"}
        )
        with urlopen(req, timeout=self.timeout_seconds) as resp:  # noqa: S310
            return json.loads(resp.read().decode())

    def list_bounties(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List open bounty postings on the Widespread board.

        Args:
            limit: max number of bounties to return.
        """
        state = self._get("/api/state")
        return (state.get("bounties") or [])[:limit]

    def list_markets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List open escrowed work markets backed by NEAR escrow.

        Args:
            limit: max number of markets to return.
        """
        markets = self._get("/api/markets")
        items = markets.get("markets", markets) if isinstance(markets, dict) else markets
        return (items or [])[:limit]

    def feed_search(self, query: str = "", source: str = "", limit: int = 25) -> List[Dict[str, Any]]:
        """Search the cross-forge work feed (GitHub/GitLab/Forgejo/Tangled/
        Radicle, bounty platforms, agent registries).

        Args:
            query: free-text filter.
            source: platform filter (e.g. "github", "gitlab").
            limit: max items.
        """
        q = f"?q={quote(query)}&source={quote(source)}&limit={limit}"
        doc = self._get(f"/api/feed{q}")
        items = doc.get("items", doc) if isinstance(doc, dict) else doc
        return (items or [])[:limit]

    def get_feed_item(self, item_id: str) -> Dict[str, Any]:
        """Get one feed item by id.

        Args:
            item_id: feed item identifier from feed_search results.
        """
        return self._get(f"/api/feed/{quote(item_id)}")

    def list_runners(self) -> List[Dict[str, Any]]:
        """List registered sealed execution runners (browser/WASM/microVM
        compartments — untrusted repo code never runs bare)."""
        runners = self._get("/api/runners")
        items = runners.get("runners", runners) if isinstance(runners, dict) else runners
        return items or []

    def get_state(self) -> Dict[str, Any]:
        """Get the live claims + bounties dashboard snapshot (includes
        adjudication verdicts and payout traces)."""
        return self._get("/api/state")
