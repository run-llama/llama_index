from datetime import datetime, timedelta, timezone

import pytest

from llama_index.tools.mcp import ScreenContextConnector, normalize_screen_context

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def payload(*records):
    return {
        "source": "observed_screen",
        "trust": "untrusted",
        "records": [
            {
                "source": "observed_screen",
                "trust": "untrusted",
                "frame_id": frame_id,
                "ts": (NOW + timedelta(minutes=offset)).timestamp(),
                "app": "Browser",
                "title": title,
                "text": text,
                "domains": ["example.com"],
            }
            for frame_id, offset, title, text in records
        ],
    }


def test_normalize_filters_the_explicit_upper_bound():
    result = normalize_screen_context(
        payload(("inside", -5, "Docs", "in range"), ("outside", 5, "Docs", "future")),
        start=NOW - timedelta(minutes=10),
        end=NOW,
    )
    assert [item.frame_id for item in result] == ["inside"]
    assert result[0].trust == "untrusted"
    assert result[0].observed_at.tzinfo == timezone.utc


def test_normalize_rejects_missing_trust_marker():
    data = payload(("frame", -1, "Docs", "text"))
    data["records"][0]["trust"] = "trusted"
    with pytest.raises(ValueError, match="untrusted-data markers"):
        normalize_screen_context(data, start=NOW - timedelta(minutes=2), end=NOW)


@pytest.mark.asyncio
async def test_connector_passes_since_bound_and_filters_end():
    class FakeClient:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            result = type("Result", (), {})()
            result.structuredContent = payload(
                ("frame", -1, "Docs", "text"), ("future", 1, "Docs", "future")
            )
            return result

    client = FakeClient()
    excerpts = await ScreenContextConnector(client).search(
        "text", start=NOW - timedelta(minutes=2), end=NOW, now=NOW
    )
    assert [item.frame_id for item in excerpts] == ["frame"]
    assert client.calls == [
        (
            "search_screen_history",
            {"query": "text", "limit": 10, "since_minutes": 2},
        )
    ]
