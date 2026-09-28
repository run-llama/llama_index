from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.tools.mcp import ScreenContextReader


class FakeScreenContextClient:
    def __init__(self) -> None:
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if arguments.get("cursor") is None:
            payload = {
                "records": [
                    {
                        "frame_id": "frame-before",
                        "observed_at": 1780001000,
                        "text": "The agent error is before the requested range.",
                    }
                ],
                "next_cursor": "page-2",
            }
        else:
            payload = {
                "records": [
                    {
                        "frame_id": "frame-match",
                        "observed_at": 1780003800,
                        "app": "Browser",
                        "app_bundle": "browser",
                        "title": "Issue tracker",
                        "text": "The agent error is visible here.",
                        "domains": ["example.com"],
                    },
                    {
                        "frame_id": "frame-after",
                        "observed_at": 1780007400,
                        "app": "Editor",
                        "app_bundle": "editor",
                        "title": "Source",
                        "text": "The agent error is outside the requested range.",
                        "domains": [],
                    },
                ],
                "next_cursor": None,
            }
        return SimpleNamespace(structuredContent=payload, content=[], isError=False)


def test_reader_implements_base_reader():
    assert issubclass(ScreenContextReader, BaseReader)


@pytest.mark.asyncio
async def test_reader_paginates_and_filters_exact_time_range():
    client = FakeScreenContextClient()
    reader = ScreenContextReader(client)

    documents = await reader.aload_data(
        query=" AGENT ERROR ",
        start_time=datetime.fromtimestamp(1780002000, timezone.utc),
        end_time=datetime.fromtimestamp(1780005600, timezone.utc),
    )

    assert len(documents) == 1
    assert documents[0].text == "The agent error is visible here."
    assert documents[0].metadata["source"] == "observed_screen"
    assert documents[0].metadata["trust"] == "untrusted"
    assert documents[0].metadata["app"] == "Browser"
    assert documents[0].metadata["window_title"] == "Issue tracker"
    assert [call[0] for call in client.calls] == [
        "get_day_material",
        "get_day_material",
    ]
    assert client.calls[1][1]["cursor"] == "page-2"


def test_reader_load_data_supports_sync_call():
    client = FakeScreenContextClient()
    reader = ScreenContextReader(client)

    documents = reader.load_data(
        query="agent",
        start_time=datetime.fromtimestamp(1780002000, timezone.utc),
        end_time=datetime.fromtimestamp(1780005600, timezone.utc),
    )

    assert len(documents) == 1


def test_reader_rejects_invalid_range():
    reader = ScreenContextReader(FakeScreenContextClient())

    with pytest.raises(ValueError, match="end_time must be after start_time"):
        reader.load_data(
            query="agent",
            start_time=datetime.fromtimestamp(1780005600, timezone.utc),
            end_time=datetime.fromtimestamp(1780002000, timezone.utc),
        )


def test_reader_rejects_non_integer_limit():
    reader = ScreenContextReader(FakeScreenContextClient())

    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        reader.load_data(
            query="agent",
            start_time=datetime.fromtimestamp(1780002000, timezone.utc),
            limit=2.5,
        )
