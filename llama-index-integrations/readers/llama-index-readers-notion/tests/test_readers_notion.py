from llama_index.core.readers.base import BaseReader
from llama_index.readers.notion import NotionPageReader


def test_class():
    names_of_base_classes = [b.__name__ for b in NotionPageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_query_database_does_not_leak_cursor_between_calls(monkeypatch):
    """A paginated database must not set the next call's starting cursor.

    `query_database` writes "start_cursor" into `query_dict` while paginating. With a
    mutable default argument that dict is created once and shared by every call that
    relies on the default, so a database that paginates leaves its last cursor behind
    and the NEXT database read starts from it. `load_data(database_ids=[...])` calls
    `query_database(database_id)` without a `query_dict`, so a single call reading two
    databases is enough to trigger it.
    """
    reader = NotionPageReader(integration_token="secret")

    sent_bodies = []
    # First database paginates once; every later request is a single page.
    responses = [
        {
            "results": [{"id": "a1"}],
            "has_more": True,
            "next_cursor": "CURSOR-FROM-DB-A",
        },
        {"results": [{"id": "a2"}], "has_more": False},
        {"results": [{"id": "b1"}], "has_more": False},
    ]

    def fake_request(method, url, headers=None, json=None, **kwargs):
        sent_bodies.append(dict(json))
        return _FakeResponse(responses[len(sent_bodies) - 1])

    monkeypatch.setattr(reader, "_request_with_retry", fake_request)

    assert reader.query_database("db-a") == ["a1", "a2"]
    assert reader.query_database("db-b") == ["b1"]

    # The third request is the first one made for db-b. It must start at the beginning.
    first_request_for_db_b = sent_bodies[2]
    assert "start_cursor" not in first_request_for_db_b, (
        f"db-b's first query carried db-a's pagination cursor: {first_request_for_db_b}"
    )
    assert first_request_for_db_b == {"page_size": 100}


def test_query_database_does_not_mutate_caller_dict(monkeypatch):
    """Pagination state must not be written into a dict the caller still owns."""
    reader = NotionPageReader(integration_token="secret")

    responses = [
        {"results": [{"id": "p1"}], "has_more": True, "next_cursor": "CURSOR"},
        {"results": [{"id": "p2"}], "has_more": False},
    ]
    calls = {"n": 0}

    def fake_request(method, url, headers=None, json=None, **kwargs):
        payload = responses[calls["n"]]
        calls["n"] += 1
        return _FakeResponse(payload)

    monkeypatch.setattr(reader, "_request_with_retry", fake_request)

    caller_dict = {"page_size": 50}
    reader.query_database("db-a", query_dict=caller_dict)

    assert caller_dict == {"page_size": 50}
