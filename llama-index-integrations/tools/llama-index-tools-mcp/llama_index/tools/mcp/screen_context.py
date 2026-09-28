"""
Explicit, read-only access to a local ScreenContextAgent MCP server.

ScreenContextAgent returns OCR observations, not trusted instructions. This
wrapper keeps the local server private, exposes only its search tool to an
agent, and makes the requested time window explicit before results enter a
LlamaIndex workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, isfinite
from typing import Any, Mapping

from mcp.client.session import ClientSession
from llama_index.tools.mcp.base import McpToolSpec


@dataclass(frozen=True)
class ScreenContextExcerpt:
    """One untrusted OCR observation returned by ScreenContextAgent."""

    frame_id: str
    observed_at: datetime
    app: str
    title: str
    text: str
    domains: tuple[str, ...]
    source: str = "observed_screen"
    trust: str = "untrusted"


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _payload(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping) and "records" in result:
        return result
    for attribute in ("structuredContent", "structured_content"):
        value = getattr(result, attribute, None)
        if isinstance(value, Mapping):
            return value
    raise ValueError("ScreenContextAgent response has no structured content")


def normalize_screen_context(
    result: Any,
    *,
    start: datetime,
    end: datetime,
) -> list[ScreenContextExcerpt]:
    """
    Validate and bound a ScreenContextAgent search response.

    The server's ``since_minutes`` parameter is an inclusive lower bound, so
    the client applies the explicit upper bound locally as well.
    """
    start_utc, end_utc = _utc(start, "start"), _utc(end, "end")
    if start_utc >= end_utc:
        raise ValueError("start must be before end")

    payload = _payload(result)
    if (
        payload.get("source") != "observed_screen"
        or payload.get("trust") != "untrusted"
    ):
        raise ValueError(
            "ScreenContextAgent response is missing untrusted-data markers"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("ScreenContextAgent response records must be a list")

    excerpts: list[ScreenContextExcerpt] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("ScreenContextAgent record must be an object")
        if (
            record.get("source") != "observed_screen"
            or record.get("trust") != "untrusted"
        ):
            raise ValueError(
                "ScreenContextAgent record is missing untrusted-data markers"
            )
        frame_id, app, title, text = (
            record.get("frame_id"),
            record.get("app"),
            record.get("title"),
            record.get("text"),
        )
        timestamp = record.get("ts")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("ScreenContextAgent record has no frame_id")
        if not all(isinstance(value, str) for value in (app, title, text)):
            raise ValueError("ScreenContextAgent record metadata must be text")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not isfinite(timestamp)
        ):
            raise ValueError("ScreenContextAgent record has an invalid timestamp")
        observed_at = datetime.fromtimestamp(timestamp, timezone.utc)
        if not start_utc <= observed_at < end_utc:
            continue
        domains = record.get("domains", [])
        if not isinstance(domains, list) or not all(
            isinstance(domain, str) for domain in domains
        ):
            raise ValueError("ScreenContextAgent record domains must be a list of text")
        excerpts.append(
            ScreenContextExcerpt(
                frame_id=frame_id,
                observed_at=observed_at,
                app=app,
                title=title,
                text=text,
                domains=tuple(domains),
            )
        )
    return excerpts


class ScreenContextConnector:
    """A bounded connector for the local ScreenContextAgent MCP server."""

    def __init__(self, client: ClientSession) -> None:
        self.client = client

    def as_tool_spec(self) -> McpToolSpec:
        """Expose only explicit, read-only screen-history search to an agent."""
        return McpToolSpec(client=self.client, allowed_tools=["search_screen_history"])

    async def search(
        self,
        query: str,
        *,
        start: datetime,
        end: datetime,
        limit: int = 10,
        now: datetime | None = None,
    ) -> list[ScreenContextExcerpt]:
        """Search a caller-selected time window, returning OCR as data."""
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must contain 1–500 characters")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 50
        ):
            raise ValueError("limit must be between 1 and 50")
        start_utc, end_utc = _utc(start, "start"), _utc(end, "end")
        if start_utc >= end_utc:
            raise ValueError("start must be before end")
        now_utc = _utc(now or datetime.now(timezone.utc), "now")
        if start_utc > now_utc:
            raise ValueError("start cannot be in the future")
        since_minutes = max(1, ceil((now_utc - start_utc).total_seconds() / 60))
        result = await self.client.call_tool(
            "search_screen_history",
            {"query": query.strip(), "limit": limit, "since_minutes": since_minutes},
        )
        return normalize_screen_context(result, start=start_utc, end=end_utc)
