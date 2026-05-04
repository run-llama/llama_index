"""
Repull tool spec.

Wraps the typed [`repull-sdk`](https://pypi.org/project/repull-sdk/) Python client
so an agent can list + inspect vacation-rental data (properties, reservations,
markets, conversations) and kick off OAuth onboarding for new channel accounts.

Read-only customer-facing surface only. Admin / billing / superadmin endpoints
are intentionally excluded.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.tools.tool_spec.base import BaseToolSpec


DEFAULT_BASE_URL = "https://api.repull.dev"


class RepullToolSpec(BaseToolSpec):
    """
    Tool spec for the Repull vacation-rental API (https://api.repull.dev).

    Repull is a unified API in front of 50+ property-management systems and the
    Airbnb / Booking.com / VRBO / Plumguide channels. This spec exposes a small,
    high-leverage subset of the SDK surface for agent use:

    * Properties — list and get
    * Reservations — list with filters
    * Markets — list customer markets, search the discovery catalog, deep-dive
      one market for pricing comp data
    * Conversations — list message threads
    * Connect — kick off white-label OAuth onboarding for a new channel account

    Get an API key at https://repull.dev/dashboard. Sandbox keys start with
    ``sk_test_``, live keys with ``sk_live_``.

    Example:
        >>> from llama_index.tools.repull import RepullToolSpec
        >>> from llama_index.core.agent.workflow import FunctionAgent
        >>> from llama_index.llms.openai import OpenAI
        >>>
        >>> repull = RepullToolSpec(api_key="sk_live_...")
        >>> agent = FunctionAgent(
        ...     tools=repull.to_tool_list(),
        ...     llm=OpenAI(model="gpt-4.1"),
        ... )
        >>> await agent.run("How many active properties do I have?")

    """

    spec_functions = [
        "list_properties",
        "get_property",
        "list_reservations",
        "list_markets",
        "search_markets",
        "get_market",
        "list_conversations",
        "create_connect_session",
    ]

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the Repull client.

        Args:
            api_key: Your Repull API key (``sk_test_*`` or ``sk_live_*``).
                Get one at https://repull.dev/dashboard.
            base_url: API base URL. Defaults to https://api.repull.dev. Override
                for self-hosted deployments.
            timeout: Per-request timeout in seconds. Defaults to 30.

        """
        # Lazy import so a missing repull-sdk only blows up at instantiation,
        # not at module import — matches the pattern used by other LlamaIndex
        # tool specs (e.g. Shopify, Salesforce).
        from repull import AuthenticatedClient

        self._client = AuthenticatedClient(
            base_url=base_url,
            token=api_key,
            timeout=timeout,
            raise_on_unexpected_status=False,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_dict(obj: Any) -> Any:
        """
        Convert an attrs-based response object from ``repull-sdk`` into plain
        Python primitives so the LLM gets something it can read back.

        Falls through unchanged if the value is ``None`` or already a primitive.
        """
        if obj is None:
            return None
        # repull-sdk models all expose `.to_dict()` (attrs + openapi-python-client)
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if isinstance(obj, list):
            return [RepullToolSpec._to_dict(item) for item in obj]
        if isinstance(obj, dict):
            return {k: RepullToolSpec._to_dict(v) for k, v in obj.items()}
        return obj

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    def list_properties(
        self,
        limit: int = 50,
        cursor: str | None = None,
        status: str | None = None,
    ) -> Any:
        """
        List properties (listings) belonging to the authenticated workspace.

        Use this when the user asks things like "what properties do I have",
        "list my listings", or "how many active properties am I managing".

        Args:
            limit: Page size, 1–100. Defaults to 50.
            cursor: Opaque pagination cursor returned by the previous page's
                ``pagination.next_cursor``. Omit on the first call.
            status: Filter by property status. One of ``"active"`` or
                ``"all"``. Omit for the API default.

        Returns:
            Dict with ``data`` (list of properties) and ``pagination``
            (``has_more``, ``next_cursor``, ``total``).

        """
        from repull.api.properties import list_properties as ep
        from repull.models.list_properties_status import ListPropertiesStatus
        from repull.types import UNSET

        status_arg: Any = UNSET
        if status is not None:
            status_arg = ListPropertiesStatus(status)

        result = ep.sync(
            client=self._client,
            limit=limit,
            cursor=cursor if cursor is not None else UNSET,
            status=status_arg,
        )
        return self._to_dict(result)

    def get_property(self, property_id: int) -> Any:
        """
        Fetch a single property by its Repull id.

        Use this when you already know the numeric property id (for example
        from a previous ``list_properties`` call) and need full details: name,
        address, bedroom / bathroom counts, channel mappings, status.

        Args:
            property_id: Repull property id. Workspace-scoped — an id from one
                workspace is not valid in another.

        Returns:
            Property dict, or ``None`` if not found in this workspace.

        """
        from repull.api.properties import get_property as ep

        result = ep.sync(id=property_id, client=self._client)
        return self._to_dict(result)

    # ------------------------------------------------------------------ #
    # Reservations
    # ------------------------------------------------------------------ #

    def list_reservations(
        self,
        limit: int = 50,
        cursor: str | None = None,
        platform: str | None = None,
        status: str | None = None,
        listing_id: int | None = None,
        check_in_after: str | None = None,
        check_in_before: str | None = None,
    ) -> Any:
        """
        List reservations across the workspace, with agent-friendly filters.

        Use this to answer questions like "show my Airbnb reservations next
        week", "any cancellations this month", or "list reservations for
        property 1234".

        Args:
            limit: Page size, 1–100. Defaults to 50.
            cursor: Opaque pagination cursor from ``pagination.next_cursor``.
            platform: Restrict to one channel. Common values: ``"airbnb"``,
                ``"booking.com"``, ``"vrbo"``, ``"website"``, ``"manual"``.
            status: Filter by reservation status. One of ``"confirmed"``,
                ``"pending"``, ``"completed"``, ``"cancelled"``.
            listing_id: Restrict to a single property (Repull listing id).
            check_in_after: ISO date (``YYYY-MM-DD``). Only reservations
                checking in on or after this date.
            check_in_before: ISO date (``YYYY-MM-DD``). Only reservations
                checking in on or before this date.

        Returns:
            Dict with ``data`` (list of reservations) and ``pagination``.

        """
        import datetime as _dt

        from repull.api.reservations import list_reservations as ep
        from repull.models.list_reservations_status import ListReservationsStatus
        from repull.types import UNSET

        def _parse_date(v: str | None) -> Any:
            if v is None:
                return UNSET
            return _dt.date.fromisoformat(v)

        status_arg: Any = UNSET
        if status is not None:
            status_arg = ListReservationsStatus(status)

        result = ep.sync(
            client=self._client,
            limit=limit,
            cursor=cursor if cursor is not None else UNSET,
            platform=platform if platform is not None else UNSET,
            status=status_arg,
            listing_id=listing_id if listing_id is not None else UNSET,
            check_in_after=_parse_date(check_in_after),
            check_in_before=_parse_date(check_in_before),
        )
        return self._to_dict(result)

    # ------------------------------------------------------------------ #
    # Markets
    # ------------------------------------------------------------------ #

    def list_markets(self) -> Any:
        """
        List the markets (cities) the customer operates in, with KPIs.

        Returns per-city stats: market share, ADR vs market, occupancy,
        ratings — plus a lightweight discovery summary of top featured markets.

        Use this to answer "what cities am I in", "where's my best market", or
        "where am I underperforming the market".

        Returns:
            Dict with ``markets`` (list of city KPI rows) and ``browse``
            (top-50 featured markets summary).

        """
        from repull.api.markets import list_markets as ep

        result = ep.sync(client=self._client)
        return self._to_dict(result)

    def search_markets(
        self,
        q: str | None = None,
        country: str | None = None,
        min_listings: int = 5,
        cursor: str | None = None,
        limit: int = 30,
        sort: str = "listings_desc",
    ) -> Any:
        """
        Search the global discovery catalog of markets across every city
        Repull tracks. Paginated.

        Use this to explore new markets the customer doesn't yet operate in —
        "find vacation-rental markets in Colorado", "biggest beach markets in
        Florida", "top markets in Spain by listing count".

        Args:
            q: Free-text search over city / country names.
            country: ISO-style country filter (e.g. ``"US"``, ``"ES"``).
            min_listings: Minimum tracked listings per market. Defaults to 5
                to filter out empty markets.
            cursor: Opaque pagination cursor.
            limit: Page size, 1–100. Defaults to 30.
            sort: Sort order. One of ``"listings_desc"`` (default) or
                ``"name_asc"``.

        Returns:
            Dict with ``data`` (list of market summaries) and ``pagination``.

        """
        from repull.api.markets import list_market_browse as ep
        from repull.models.list_market_browse_sort import ListMarketBrowseSort
        from repull.types import UNSET

        result = ep.sync(
            client=self._client,
            q=q if q is not None else UNSET,
            country=country if country is not None else UNSET,
            min_listings=min_listings,
            cursor=cursor if cursor is not None else UNSET,
            limit=limit,
            sort=ListMarketBrowseSort(sort),
        )
        return self._to_dict(result)

    def get_market(self, city: str, comps_page: int = 1) -> Any:
        """
        Deep-dive one market for pricing comp data.

        Returns price distribution, bedroom mix, property types, upcoming
        events, demand signals, monthly benchmarks, and proximity-sorted top
        comparable listings.

        Use this for pricing questions: "what should I charge in Miami next
        weekend", "show me comps for Aspen", "what's the ADR distribution in
        Lisbon".

        Args:
            city: City name (e.g. ``"Aspen"``, ``"Lisbon"``,
                ``"Radium Hot Springs"``). Repull does fuzzy matching.
            comps_page: 1-indexed page of the comps table. Defaults to 1.

        Returns:
            Dict with the full market detail payload.

        """
        from repull.api.markets import get_market as ep

        result = ep.sync(city=city, client=self._client, comps_page=comps_page)
        return self._to_dict(result)

    # ------------------------------------------------------------------ #
    # Conversations
    # ------------------------------------------------------------------ #

    def list_conversations(
        self,
        limit: int = 20,
        cursor: str | None = None,
        platform: str | None = None,
        status: str | None = None,
    ) -> Any:
        """
        List guest message threads owned by the workspace.

        Use this to surface recent conversations or filter to a single channel
        — "show my open Airbnb threads", "any unread messages today".

        Args:
            limit: Page size, 1–100. Defaults to 20.
            cursor: Opaque pagination cursor.
            platform: Restrict to one channel. One of ``"airbnb"``,
                ``"booking"``, ``"vrbo"``, ``"website"``, ``"email"``.
            status: Filter by thread status. One of ``"open"`` or
                ``"archived"``.

        Returns:
            Dict with ``data`` (list of conversation summaries) and
            ``pagination``.

        """
        from repull.api.conversations import list_conversations as ep
        from repull.models.list_conversations_platform import (
            ListConversationsPlatform,
        )
        from repull.models.list_conversations_status import ListConversationsStatus
        from repull.types import UNSET

        platform_arg: Any = UNSET
        if platform is not None:
            platform_arg = ListConversationsPlatform(platform)

        status_arg: Any = UNSET
        if status is not None:
            status_arg = ListConversationsStatus(status)

        result = ep.sync(
            client=self._client,
            limit=limit,
            cursor=cursor if cursor is not None else UNSET,
            platform=platform_arg,
            status=status_arg,
        )
        return self._to_dict(result)

    # ------------------------------------------------------------------ #
    # Connect (OAuth onboarding)
    # ------------------------------------------------------------------ #

    def create_connect_session(
        self,
        redirect_url: str,
        allowed_providers: list[str] | None = None,
    ) -> Any:
        """
        Mint a white-label Connect session URL the user can visit to link a
        new channel account (Airbnb, Booking.com, VRBO, etc.) via OAuth.

        Use this when the user says things like "connect my Airbnb",
        "onboard a new Booking.com property", or "add another channel". The
        agent should return the session URL and tell the user to open it in
        their browser.

        Args:
            redirect_url: Where to send the user after they finish the OAuth
                flow. Must be an HTTPS URL.
            allowed_providers: Optional whitelist of provider keys to show in
                the picker. Common values: ``["airbnb"]``, ``["booking.com"]``,
                ``["airbnb", "vrbo"]``. Omit to show every supported provider.

        Returns:
            Dict containing the session ``url`` (open in browser) and
            ``session_id``. After the user completes the flow, poll
            ``GET /v1/connect/{provider}`` to confirm.

        """
        from repull.api.connect import create_connect_session as ep
        from repull.models.create_connect_session_body import (
            CreateConnectSessionBody,
        )
        from repull.types import UNSET

        body = CreateConnectSessionBody(
            redirect_url=redirect_url,
            allowed_providers=allowed_providers
            if allowed_providers is not None
            else UNSET,
        )
        result = ep.sync(client=self._client, body=body)
        return self._to_dict(result)
