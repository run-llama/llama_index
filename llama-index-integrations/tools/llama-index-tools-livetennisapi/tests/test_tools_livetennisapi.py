import httpx
import respx
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.livetennisapi import LiveTennisAPIToolSpec

BASE = "https://api.livetennisapi.com/api/public/v1"


def test_class():
    names_of_base_classes = [b.__name__ for b in LiveTennisAPIToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_to_tool_list():
    spec = LiveTennisAPIToolSpec(api_key="twjp_test")
    tools = spec.to_tool_list()
    names = {t.metadata.name for t in tools}
    assert "get_live_matches" in names
    assert "check_status" in names
    # The docstring is what the LLM reads, so it must become the description.
    live = next(t for t in tools if t.metadata.name == "get_live_matches")
    assert "currently in progress" in live.metadata.description


@respx.mock
def test_get_live_matches_smoke():
    respx.get(f"{BASE}/matches").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": 42,
                        "tournament": "ATP Test Open",
                        "round": "SF",
                        "surface": "Hard",
                        "status": "live",
                        "players": {
                            "p1": {"id": 1, "name": "Player One"},
                            "p2": {"id": 2, "name": "Player Two"},
                        },
                        "score": {
                            "sets": [1, 0],
                            "games": [[6, 2], [4, 1]],
                            "points": ["30", "15"],
                            "server": 1,
                        },
                    }
                ],
                "meta": {"limit": 20, "offset": 0, "count": 1},
            },
        )
    )

    spec = LiveTennisAPIToolSpec(api_key="twjp_test")
    out = spec.get_live_matches(limit=20)
    assert "1 live match(es)" in out
    assert "[42] Player One vs Player Two" in out
    assert "ATP Test Open" in out
    assert "6-4 2-1" in out
    assert "Serving: player 1" in out


@respx.mock
def test_get_live_matches_empty():
    respx.get(f"{BASE}/matches").mock(
        return_value=httpx.Response(200, json={"data": [], "meta": {"count": 0}})
    )
    spec = LiveTennisAPIToolSpec(api_key="twjp_test")
    assert spec.get_live_matches() == "No matches are live right now."


@respx.mock
def test_paid_tool_returns_tier_message():
    # A FREE key hitting a PRO endpoint gets a 403 upgrade_required; the tool
    # must turn that into a plain-English message, not raise.
    respx.get(f"{BASE}/markets/42/prices").mock(
        return_value=httpx.Response(403, json={"error": "upgrade_required"})
    )
    spec = LiveTennisAPIToolSpec(api_key="twjp_test")
    out = spec.get_match_odds(42)
    assert "PRO plan" in out
    assert "livetennisapi.com" in out


@respx.mock
def test_search_players_smoke():
    respx.get(f"{BASE}/players").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": 7,
                        "name": "Carlos Alcaraz",
                        "country": "ESP",
                        "ranking": 2,
                        "tour": "ATP",
                    }
                ]
            },
        )
    )
    spec = LiveTennisAPIToolSpec(api_key="twjp_test")
    out = spec.search_players("alcaraz")
    assert "[7] Carlos Alcaraz (ESP) - rank 2 - ATP" in out
