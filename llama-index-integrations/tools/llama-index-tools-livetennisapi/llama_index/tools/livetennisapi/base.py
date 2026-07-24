"""Live Tennis API tool spec.

Wraps the official ``livetennisapi`` Python client (``pip install livetennisapi``)
and exposes it as a LlamaIndex :class:`BaseToolSpec` so an agent can read live
tennis scores, players, fixtures and results.

The methods mirror the Live Tennis API's official MCP server. Every method
returns a plain, human-readable string rather than raw JSON, because the string
is what the LLM reads. Errors (a bad id, a rate limit, an endpoint that needs a
higher plan) are turned into a clear sentence instead of an exception, so a tool
call never crashes the agent loop.

The free tier covers live/upcoming matches, scores, players, fixtures and the
status check. The higher-tier reads (recent results, match events, market odds,
model analysis) are exposed too, but on a free key they return a short
plain-English message pointing at the plan that unlocks them.
"""

from __future__ import annotations

from typing import Any, List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

_PRICING_URL = "https://livetennisapi.com/#pricing"


class LiveTennisAPIToolSpec(BaseToolSpec):
    """Live Tennis API tool spec.

    Real-time tennis scores, players, rankings, fixtures and results for ATP,
    WTA, Challenger and ITF, from the Live Tennis API.

    Args:
        api_key (Optional[str]): Your Live Tennis API key. When omitted the
            client falls back to the ``LIVETENNISAPI_KEY`` environment variable.
            Get a free key (1000 requests/day) at
            https://livetennisapi.com/subscribe/free.
        auth_header (str): ``"bearer"`` (default) sends
            ``Authorization: Bearer <key>``; ``"x-api-key"`` sends
            ``X-API-Key: <key>``. Both are accepted by the API.
    """

    spec_functions = [
        "get_live_matches",
        "get_upcoming_matches",
        "get_match",
        "search_players",
        "get_player",
        "get_fixtures",
        "get_recent_results",
        "get_match_events",
        "get_match_odds",
        "get_match_analysis",
        "check_status",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        auth_header: str = "bearer",
    ) -> None:
        """Initialize with parameters."""
        try:
            from livetennisapi import LiveTennisAPI
        except ImportError:
            raise ImportError(
                "The Live Tennis API tool requires the livetennisapi package to be "
                "installed. Please install it using `pip install livetennisapi`."
            )

        self._client = LiveTennisAPI(api_key=api_key, auth_header=auth_header)

    # -- error handling -------------------------------------------------------

    def _guard(self, fn: Any) -> str:
        """Run a client call, turning the API's errors into plain sentences.

        A higher-tier endpoint on a lower plan raises ``UpgradeRequired``; that
        becomes a message naming the plan that unlocks it rather than an
        exception the agent has to handle.
        """
        from livetennisapi import (
            APIConnectionError,
            LiveTennisAPIError,
            NotFound,
            RateLimited,
            Unauthorized,
            UpgradeRequired,
        )

        try:
            return fn()
        except UpgradeRequired as exc:
            tier = getattr(exc, "required_tier", None)
            plan = f"the {tier} plan" if tier else "a higher plan"
            return (
                f"This data requires {plan}. Your current API key does not unlock it. "
                f"See {_PRICING_URL} to upgrade."
            )
        except Unauthorized:
            return (
                "The Live Tennis API rejected the key (unauthorized). Check that a "
                "valid key is set, or get a free one at "
                "https://livetennisapi.com/subscribe/free."
            )
        except RateLimited as exc:
            wait = getattr(exc, "retry_after", None)
            hint = f" Retry in about {wait:g}s." if wait else ""
            return f"Rate limit reached for the current plan.{hint}"
        except NotFound:
            return "No data was found for that request."
        except APIConnectionError:
            return "Could not reach the Live Tennis API. Please try again shortly."
        except LiveTennisAPIError as exc:
            return f"The Live Tennis API returned an error: {exc}"

    # -- formatting -----------------------------------------------------------

    @staticmethod
    def _names(match: Any) -> tuple[str, str]:
        """Return ``(player1, player2)`` display names for a match."""
        players = getattr(match, "players", None) or {}
        p1 = players.get("p1") if isinstance(players, dict) else None
        p2 = players.get("p2") if isinstance(players, dict) else None
        n1 = (getattr(p1, "name", None) if p1 is not None else None) or "?"
        n2 = (getattr(p2, "name", None) if p2 is not None else None) or "?"
        return n1, n2

    @staticmethod
    def _format_score(score: Any) -> str:
        """Render a :class:`Score` as ``6-4 3-6 2-1`` (set by set)."""
        if score is None:
            return "-"
        games = getattr(score, "games", None)
        if not games or len(games) < 2:
            sets = getattr(score, "sets", None)
            return "-".join(str(s) for s in sets) if sets else "-"
        p1, p2 = games[0] or [], games[1] or []
        parts = []
        for i in range(max(len(p1), len(p2))):
            a = p1[i] if i < len(p1) else "-"
            b = p2[i] if i < len(p2) else "-"
            parts.append(f"{a}-{b}")
        line = " ".join(parts) if parts else "-"
        points = getattr(score, "points", None)
        if points and len(points) >= 2 and not getattr(score, "is_tiebreak", False):
            line += f" ({points[0]}-{points[1]})"
        return line

    def _summarise(self, match: Any) -> str:
        """One block of text describing a match, ready for the LLM."""
        n1, n2 = self._names(match)
        mid = getattr(match, "id", None)
        tournament = getattr(match, "tournament", None) or "Unknown event"
        rnd = getattr(match, "round", None)
        surface = getattr(match, "surface", None)
        status = getattr(match, "status", None) or getattr(match, "event_status", None)
        header = f"[{mid}] {n1} vs {n2}"
        meta = tournament
        if rnd:
            meta += f", {rnd}"
        if surface:
            meta += f" ({surface})"
        lines = [header, f"  {meta}"]
        if status:
            lines.append(f"  Status: {status}")
        score = getattr(match, "score", None)
        if score is not None:
            lines.append(f"  Score: {self._format_score(score)}")
            server = getattr(score, "server", None)
            if server:
                lines.append(f"  Serving: player {server}")
            if getattr(score, "is_tiebreak", None):
                lines.append("  In a tiebreak")
            wp = getattr(score, "win_probability_p1", None)
            if wp is not None:
                lines.append(f"  Model win prob (player 1): {wp * 100:.1f}%")
        winner = getattr(match, "winner", None)
        if winner:
            lines.append(f"  Winner: player {winner} ({n1 if winner == 1 else n2})")
        return "\n".join(lines)

    def _format_player(self, p: Any) -> str:
        """A one-line player summary: ``[id] Name (COUNTRY) — rank N · TOUR``."""
        pid = getattr(p, "id", None)
        name = getattr(p, "name", None) or "?"
        country = getattr(p, "country", None)
        ranking = getattr(p, "ranking", None)
        tour = getattr(p, "tour", None)
        line = f"[{pid}] {name}"
        if country:
            line += f" ({country})"
        if ranking is not None:
            line += f" - rank {ranking}"
        if tour:
            line += f" - {tour}"
        return line

    # -- FREE tier ------------------------------------------------------------

    def get_live_matches(self, limit: int = 20) -> str:
        """
        List tennis matches currently in progress, with live scores.

        Covers ATP, WTA, Challenger and ITF. Use this for "what tennis is on
        right now". Each match line begins with its numeric id in brackets,
        which you can pass to get_match, get_match_odds or get_match_analysis.

        Args:
            limit (int): Maximum number of matches to return (1-200). Default 20.

        """

        def run() -> str:
            page = self._client.list_matches(status="live", limit=limit)
            if not len(page):
                return "No matches are live right now."
            body = "\n\n".join(self._summarise(m) for m in page)
            return f"{len(page)} live match(es):\n\n{body}"

        return self._guard(run)

    def get_upcoming_matches(self, limit: int = 20) -> str:
        """
        List tennis matches scheduled to start soon, with players and tournament.

        Args:
            limit (int): Maximum number of matches to return (1-200). Default 20.

        """

        def run() -> str:
            page = self._client.list_matches(status="upcoming", limit=limit)
            if not len(page):
                return "No upcoming matches are scheduled."
            body = "\n\n".join(self._summarise(m) for m in page)
            return f"{len(page)} upcoming match(es):\n\n{body}"

        return self._guard(run)

    def get_match(self, match_id: int) -> str:
        """
        Full detail for one match by id: players, score, surface, round and status.

        Includes market prices on the PRO plan and model analysis on the ULTRA
        plan when your key unlocks them.

        Args:
            match_id (int): The match id, as returned by get_live_matches,
                get_upcoming_matches or get_recent_results.

        """

        def run() -> str:
            match = self._client.get_match(match_id)
            if match is None:
                return "No data was found for that match id."
            out = self._summarise(match)
            market = getattr(match, "market", None)
            if market is not None:
                out += f"\n\nMarket: {getattr(market, 'question', None) or '-'}"
                for price in getattr(market, "prices", None) or []:
                    side = getattr(price, "side", None)
                    mid = getattr(price, "mid", None)
                    bid = getattr(price, "bid", None)
                    ask = getattr(price, "ask", None)
                    out += f"\n  Side {side}: mid {mid} (bid {bid} / ask {ask})"
            analysis = getattr(match, "analysis", None)
            profile = getattr(analysis, "profile", None) if analysis is not None else None
            if profile:
                out += "\n\nModel analysis:"
                wp = profile.get("win_probability_p1")
                if wp is not None:
                    out += f"\n  Win probability (player 1): {wp * 100:.1f}%"
                factors = profile.get("key_factors")
                if factors:
                    out += f"\n  Key factors: {'; '.join(factors)}"
            return out

        return self._guard(run)

    def search_players(self, query: str, limit: int = 10) -> str:
        """
        Search tennis players by name. Returns id, country, ranking and tour.

        Use the returned id with get_player.

        Args:
            query (str): Full or partial player name, e.g. "alcaraz".
            limit (int): Maximum number of players to return (1-200). Default 10.

        """

        def run() -> str:
            page = self._client.search_players(query, limit=limit)
            if not len(page):
                return f'No players matched "{query}".'
            return "\n".join(self._format_player(p) for p in page)

        return self._guard(run)

    def get_player(self, player_id: int) -> str:
        """
        One player's profile: ranking, country, handedness, date of birth and stats.

        Args:
            player_id (int): The player id, as returned by search_players.

        """

        def run() -> str:
            p = self._client.get_player(player_id)
            if p is None:
                return "No data was found for that player id."
            rows = [f"{getattr(p, 'name', None) or 'Unknown'} [{getattr(p, 'id', None)}]"]
            country = getattr(p, "country", None)
            if country:
                rows.append(f"Country: {country}")
            ranking = getattr(p, "ranking", None)
            if ranking is not None:
                pts = getattr(p, "ranking_points", None)
                rows.append(f"Ranking: {ranking}" + (f" ({pts} pts)" if pts else ""))
            movement = getattr(p, "ranking_movement", None)
            if movement:
                rows.append(f"Movement: {movement}")
            hand = getattr(p, "hand", None)
            if hand:
                rows.append(f"Plays: {'right-handed' if hand == 'R' else 'left-handed'}")
            birthday = getattr(p, "birthday", None)
            if birthday:
                rows.append(f"Born: {birthday}")
            tour = getattr(p, "tour", None)
            if tour:
                rows.append(f"Tour: {tour}")
            stats = getattr(p, "stats", None)
            if isinstance(stats, dict) and stats:
                # Keep this LLM-friendly: show scalar stats inline, but never
                # dump the large nested rating/season trees the API returns.
                parts = []
                for k, v in stats.items():
                    if isinstance(v, (str, int, float, bool)):
                        parts.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        parts.append(f"{k}: {len(v)} entries (available)")
                    elif isinstance(v, dict):
                        parts.append(f"{k}: available")
                if parts:
                    rows.append("Cached stats - " + ", ".join(parts))
            return "\n".join(rows)

        return self._guard(run)

    def get_fixtures(self, limit: int = 20) -> str:
        """
        Upcoming scheduled tennis fixtures, earliest first — the forward schedule.

        Args:
            limit (int): Maximum number of fixtures to return (1-200). Default 20.

        """

        def run() -> str:
            page = self._client.list_fixtures(limit=limit)
            if not len(page):
                return "No upcoming fixtures."
            lines = []
            for f in page:
                date = getattr(f, "event_date", None) or "?"
                tournament = getattr(f, "tournament", None) or "?"
                rnd = getattr(f, "round", None)
                p1 = getattr(f, "player1_name", None) or "?"
                p2 = getattr(f, "player2_name", None) or "?"
                suffix = f" ({rnd})" if rnd else ""
                lines.append(f"{date} - {tournament}{suffix}: {p1} vs {p2}")
            return "\n".join(lines)

        return self._guard(run)

    def check_status(self) -> str:
        """
        Check whether the Live Tennis API is reachable and which plan the key is on.

        Useful for diagnosing why other tools are refusing data: a "requires the
        PRO plan" message from another tool is expected on a FREE or BASIC key.
        """
        from livetennisapi import NotFound, Unauthorized, UpgradeRequired

        try:
            health = self._client.health()
        except Exception as exc:  # noqa: BLE001 - report any reachability failure plainly
            return f"Could not reach the Live Tennis API: {exc}"

        status = health.get("status") if isinstance(health, dict) else None
        version = health.get("version") if isinstance(health, dict) else None
        reachable = f"API is reachable (status: {status}, version: {version})."

        if not getattr(self._client, "api_key", ""):
            return (
                f"{reachable}\nNo API key is configured, so only this check will work. "
                "Get a free key at https://livetennisapi.com/subscribe/free."
            )

        # Probe upward to discover the tier without asking the user. Only an
        # UpgradeRequired proves a tier is NOT held; a NotFound means the call
        # was allowed but that row has no data, which is evidence of entitlement.
        tier = "BASIC"
        try:
            self._client.list_matches(status="completed", limit=1)
        except Unauthorized:
            return f"{reachable}\nThe configured key was rejected (unauthorized)."

        history = None
        try:
            history = self._client.list_completed_matches(limit=1)
        except UpgradeRequired:
            tier = "FREE"

        probe_id = history[0].id if (history and len(history)) else None
        if tier != "FREE" and probe_id is not None:
            try:
                self._client.list_match_events(probe_id, limit=1)
                tier = "PRO"
            except NotFound:
                tier = "PRO"
            except UpgradeRequired:
                pass
            if tier == "PRO":
                try:
                    self._client.get_match_analysis(probe_id)
                    tier = "ULTRA"
                except NotFound:
                    tier = "ULTRA"
                except UpgradeRequired:
                    pass

        return (
            f"{reachable}\nThe configured key appears to be on the {tier} plan.\n\n"
            "FREE  = live & upcoming matches, scores, players, fixtures\n"
            "BASIC = + historical results\n"
            "PRO   = + match events and market prices\n"
            "ULTRA = + model analysis, win probability and the live feed"
        )

    # -- higher tiers (return a plain-English tier message on a lower plan) ----

    def get_recent_results(self, limit: int = 20) -> str:
        """
        Recently completed tennis matches with final scores and winners.

        Requires the BASIC plan or higher. On a FREE key this returns a short
        message naming the plan that unlocks it.

        Args:
            limit (int): Maximum number of matches to return (1-200). Default 20.

        """

        def run() -> str:
            page = self._client.list_completed_matches(limit=limit)
            if not len(page):
                return "No completed matches available."
            return "\n\n".join(self._summarise(m) for m in page)

        return self._guard(run)

    def get_match_events(self, match_id: int, limit: int = 30) -> str:
        """
        Timeline of events for a match — breaks, games won, sets won, momentum runs.

        Requires the PRO plan or higher. On a lower plan this returns a short
        message naming the plan that unlocks it.

        Args:
            match_id (int): The match id, as returned by get_live_matches.
            limit (int): Maximum number of events to return (1-200). Default 30.

        """

        def run() -> str:
            page = self._client.list_match_events(match_id, limit=limit)
            if not len(page):
                return "No events recorded for this match."
            lines = []
            for e in page:
                ts = getattr(e, "timestamp", None) or "?"
                etype = getattr(e, "type", None) or "?"
                player = getattr(e, "player", None)
                suffix = f" (player {player})" if player else ""
                lines.append(f"{ts} - {etype}{suffix}")
            return "\n".join(lines)

        return self._guard(run)

    def get_match_odds(self, match_id: int, limit: int = 10) -> str:
        """
        Match-winner market prices for a match — implied probability per player.

        Includes bid, ask and mid per side. Requires the PRO plan or higher. On
        a lower plan this returns a short message naming the plan that unlocks it.

        Args:
            match_id (int): The match id, as returned by get_live_matches.
            limit (int): Maximum number of price points to return (1-200).
                Default 10.

        """

        def run() -> str:
            market = self._client.get_market_prices(match_id, limit=limit)
            if market is None:
                return "No market data for that match."
            lines = [f"Market: {getattr(market, 'question', None) or '-'}"]
            status = getattr(market, "status", None)
            if status:
                lines.append(f"Status: {status}")
            volume = getattr(market, "volume", None)
            if volume is not None:
                lines.append(f"24h volume: {volume}")
            liquidity = getattr(market, "liquidity", None)
            if liquidity is not None:
                lines.append(f"Liquidity: {liquidity}")
            lines += ["", "Recent prices (newest first):"]
            for p in getattr(market, "prices", None) or []:
                side = getattr(p, "side", None)
                mid = getattr(p, "mid", None)
                bid = getattr(p, "bid", None)
                ask = getattr(p, "ask", None)
                ts = getattr(p, "timestamp", None)
                stamp = f" @ {ts}" if ts else ""
                lines.append(f"  side {side}: mid {mid} - bid {bid} - ask {ask}{stamp}")
            return "\n".join(lines)

        return self._guard(run)

    def get_match_analysis(self, match_id: int) -> str:
        """
        Model analysis for a match: predicted win probability, thesis, key factors.

        Requires the ULTRA plan. On a lower plan this returns a short message
        naming the plan that unlocks it.

        Args:
            match_id (int): The match id, as returned by get_live_matches.

        """

        def run() -> str:
            analysis = self._client.get_match_analysis(match_id)
            profile = getattr(analysis, "profile", None) if analysis is not None else None
            thesis = getattr(analysis, "thesis", None) if analysis is not None else None
            if analysis is None or (not profile and not thesis):
                return "No model analysis exists for this match yet."
            lines: List[str] = []
            if profile:
                lines.append("Profile:")
                wp = profile.get("win_probability_p1")
                if wp is not None:
                    lines.append(f"  Win probability (player 1): {wp * 100:.1f}%")
                closeness = profile.get("expected_closeness")
                if closeness is not None:
                    lines.append(f"  Expected closeness: {closeness}")
                volatility = profile.get("volatility_rating")
                if volatility:
                    lines.append(f"  Volatility: {volatility}")
                factors = profile.get("key_factors")
                if factors:
                    lines.append(f"  Key factors: {'; '.join(factors)}")
            if thesis:
                lines += ["", "Thesis:"]
                pick = thesis.get("pick_side")
                if pick:
                    lines.append(f"  Pick: player {pick}")
                confidence = thesis.get("confidence")
                if confidence is not None:
                    lines.append(f"  Confidence: {confidence * 100:.0f}%")
                state = thesis.get("state")
                if state:
                    lines.append(f"  State: {state}")
                reasoning = thesis.get("reasoning")
                if reasoning:
                    lines.append(f"  Reasoning: {reasoning}")
            return "\n".join(lines)

        return self._guard(run)
