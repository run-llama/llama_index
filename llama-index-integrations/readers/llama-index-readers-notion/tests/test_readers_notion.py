from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.notion import NotionPageReader


def test_class():
    names_of_base_classes = [b.__name__ for b in NotionPageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def _rich(text: str, kind: str = "text") -> Dict[str, Any]:
    item: Dict[str, Any] = {"type": kind, "plain_text": text}
    if kind == "text":
        item["text"] = {"content": text, "link": None}
    elif kind == "mention":
        item["mention"] = {"type": "date", "date": {"start": text}}
    elif kind == "equation":
        item["equation"] = {"expression": text}
    return item


def _paragraph(block_id: str, *texts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": block_id,
        "type": "paragraph",
        "has_children": False,
        "paragraph": {"rich_text": list(texts)},
    }


def _response(
    results: List[Dict[str, Any]], next_cursor: Optional[str] = None
) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {
        "results": results,
        "next_cursor": next_cursor,
        "has_more": next_cursor is not None,
    }
    return response


@pytest.fixture()
def reader() -> NotionPageReader:
    return NotionPageReader(integration_token="secret")


def test_read_block_follows_pagination_with_start_cursor(
    reader: NotionPageReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_page = [_paragraph(f"b{i}", _rich(f"line {i}")) for i in range(100)]
    second_page = [_paragraph(f"b{i}", _rich(f"line {i}")) for i in range(100, 150)]
    calls: List[Dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> MagicMock:
        # Like the Notion API: only the page has children; for blocks without
        # children (e.g. a request to /blocks/b100/children) the list is empty.
        calls.append({"url": url, "params": kwargs.get("params")})
        if url != "https://api.notion.com/v1/blocks/page-id/children":
            return _response([])
        if (kwargs.get("params") or {}).get("start_cursor") == "b100":
            return _response(second_page)
        return _response(first_page, next_cursor="b100")

    monkeypatch.setattr(reader, "_request_with_retry", fake_request)

    text = reader.read_page("page-id")

    assert text.splitlines() == [f"line {i}" for i in range(150)]
    # Both pages are requested from the page itself; the cursor is a query
    # parameter, not a block id in the URL.
    assert [c["url"] for c in calls] == [
        "https://api.notion.com/v1/blocks/page-id/children"
    ] * 2
    assert calls[1]["params"]["start_cursor"] == "b100"


def test_read_block_reads_table_rows(
    reader: NotionPageReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = {"id": "t", "type": "table", "has_children": True, "table": {}}
    rows = [
        {
            "id": f"r{i}",
            "type": "table_row",
            "has_children": False,
            "table_row": {"cells": [[_rich(a)], [_rich(b)]]},
        }
        for i, (a, b) in enumerate([("Field", "Retention"), ("Alerts", "5 years")])
    ]

    def fake_request(method: str, url: str, **kwargs: Any) -> MagicMock:
        return _response(rows if "/blocks/t/" in url else [table])

    monkeypatch.setattr(reader, "_request_with_retry", fake_request)

    assert reader.read_page("page-id").split("\n") == [
        "\tField | Retention",
        "\tAlerts | 5 years",
    ]


def test_read_block_keeps_mentions_and_equations(
    reader: NotionPageReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    block = _paragraph(
        "b",
        _rich("Due "),
        _rich("2024-05-01", "mention"),
        _rich("E = mc^2", "equation"),
    )
    monkeypatch.setattr(
        reader, "_request_with_retry", lambda *a, **k: _response([block])
    )

    assert reader.read_page("page-id").split("\n") == ["Due ", "2024-05-01", "E = mc^2"]
