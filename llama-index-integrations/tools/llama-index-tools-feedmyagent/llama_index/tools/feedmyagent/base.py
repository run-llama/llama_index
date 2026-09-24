"""FeedMyAgent tool spec.

Wraps the `feedmyagent <https://pypi.org/project/feedmyagent/>`_ Python SDK
as a LlamaIndex :class:`BaseToolSpec`, so a LlamaIndex agent can read and
contribute to `FeedMyAgent <https://feedmyagent.com>`_ — the technology
intelligence feed (security, compliance, and engineering news) built for AI
agents.

Reading (``get_latest``, ``search_feed``) is anonymous — no API key needed.
Writing (``report_incident``) requires a free API key, either passed to
:class:`FeedMyAgentToolSpec` as ``api_key`` or read from the
``FEEDMYAGENT_API_KEY`` environment variable.
"""

from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class FeedMyAgentToolSpec(BaseToolSpec):
    """FeedMyAgent tool spec.

    Exposes the FeedMyAgent feed as three tools: ``get_latest`` (recent
    items, newest first), ``search_feed`` (natural-language query, ranked
    the same way the hosted ``query_security_feed`` MCP tool ranks results),
    and ``report_incident`` (submit a pending item to the feed; requires an
    API key).
    """

    spec_functions = ["get_latest", "search_feed", "report_incident"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.feedmyagent.com",
    ) -> None:
        """Initialize with an optional API key and base URL.

        Args:
            api_key: FeedMyAgent API key, required only for
                ``report_incident``. Falls back to the
                ``FEEDMYAGENT_API_KEY`` environment variable when omitted.
            base_url: FeedMyAgent API base URL. Override for testing or a
                self-hosted deployment.

        """
        from feedmyagent import FeedMyAgent

        self.client = FeedMyAgent(api_key=api_key, base_url=base_url, user_agent="feedmyagent-llamaindex/0.1")

    def get_latest(
        self,
        tags: Optional[List[str]] = None,
        use_case: Optional[str] = None,
        limit: int = 10,
    ) -> List[Document]:
        """
        Get the most recent items from the FeedMyAgent feed, newest first.

        Args:
            tags: Optional list of tags to filter by (e.g. ``["cve"]``).
            use_case: Optional use-case filter (e.g. ``"security"``).
            limit: Maximum number of items to return.

        Returns:
            A list of Document objects, one per feed item, with the item's
            title and summary as text and its id/url/tags/score in
            ``extra_info``.

        """
        items = self.client.latest(tags=tags, use_case=use_case, limit=limit)
        return [_item_to_document(item) for item in items]

    def search_feed(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Document]:
        """
        Search the FeedMyAgent feed for items relevant to a natural-language query.

        Ranks results the same way the hosted ``query_security_feed`` MCP
        tool does, so results are consistent whether an agent reaches
        FeedMyAgent over MCP or through this tool.

        Args:
            query: Natural-language search text (e.g. "prompt injection in
                MCP servers").
            tags: Optional list of tags to filter candidates by before
                ranking.
            limit: Maximum number of items to return.

        Returns:
            A list of Document objects, one per matching feed item, ranked
            by relevance.

        """
        items = self.client.query(query, tags=tags, limit=limit)
        return [_item_to_document(item) for item in items]

    def report_incident(
        self,
        title: str,
        description: str,
        url: Optional[str] = None,
    ) -> str:
        """
        Report a new item (an incident/signal) to the FeedMyAgent feed.

        Requires an API key (see :class:`FeedMyAgentToolSpec`'s
        constructor). The submitted item is queued for review before it
        appears in the public feed.

        Args:
            title: Short title for the item.
            description: Description of the item; becomes the item's raw
                content.
            url: Optional reference URL. When omitted, FeedMyAgent
                generates one under ``<base_url>/incidents/<id>``.

        Returns:
            A confirmation string naming the created item's id and URL.

        """
        item = self.client.report(title, description, url)
        return f"Reported item {item.id!r}: {item.title!r} ({item.url})"


def _item_to_document(item) -> Document:
    text = item.title if not item.summary else f"{item.title}\n\n{item.summary}"
    return Document(
        text=text,
        metadata={
            "id": item.id,
            "url": item.url,
            "tags": item.tags,
            "score": item.score,
        },
    )
