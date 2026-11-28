import json
from typing import Any, Dict, List

import httpx
import pytest

from llama_index.readers.web import OxylabsWebAPIReader
from llama_index.readers.web.oxylabs_web_api import base
from llama_index.readers.web.oxylabs_web_api.base import OxylabsWebAPIError

PAGE = "# Page\n\nSome text."


def page(value: str = PAGE, url: str = "https://example.com/") -> Dict[str, Any]:
    return {
        "state": "done",
        "results": [
            {
                "markdown": value,
                "metadata": {
                    "url": url,
                    "statuses": {"http_request": {"code": 200, "tag": "HTTP_OK"}},
                },
            }
        ],
        "metadata": {"request_id": "42"},
    }


FAULTED = {
    "state": "faulted",
    "title": "REQUEST_FAILED_AFTER_TOO_MANY_RETRIES",
    "detail": "Try again.",
    "metadata": {"request_id": "99"},
}


class FakeWebAPI:
    """Answers requests from a queue of replies, in order; the last one repeats."""

    def __init__(self) -> None:
        self.replies: List[Any] = []
        self.requests: List[httpx.Request] = []

    def add(self, status: int, body: Any) -> "FakeWebAPI":
        self.replies.append((status, body))
        return self

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(body, Exception):
            raise body
        return httpx.Response(status, json=body)

    @property
    def paths(self) -> List[str]:
        return [f"{r.method} {r.url.path}" for r in self.requests]

    @property
    def payloads(self) -> List[Dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.content]


@pytest.fixture
def web_api(monkeypatch: pytest.MonkeyPatch) -> FakeWebAPI:
    fake = FakeWebAPI()
    transport = httpx.MockTransport(fake.handler)
    real_client, real_async_client = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(
        base.httpx, "Client", lambda **kw: real_client(transport=transport, **kw)
    )
    monkeypatch.setattr(
        base.httpx,
        "AsyncClient",
        lambda **kw: real_async_client(transport=transport, **kw),
    )
    # Retries and polling would otherwise sleep for real.
    monkeypatch.setattr(base.time, "sleep", lambda _: None)
    monkeypatch.setattr(base.random, "uniform", lambda *_: 0)
    return fake


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OXYLABS_WEB_API_KEY", "test-key")


def test_class_name() -> None:
    assert OxylabsWebAPIReader.class_name() == "OxylabsWebAPIReader"


def test_is_remote() -> None:
    assert OxylabsWebAPIReader().is_remote


class TestApiKey:
    def test_comes_from_the_environment(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, page())
        OxylabsWebAPIReader().load_data(["https://example.com"])
        assert web_api.requests[-1].headers["authorization"] == "Bearer test-key"

    def test_explicit_key_wins(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, page())
        OxylabsWebAPIReader(api_key="other").load_data(["https://example.com"])
        assert web_api.requests[-1].headers["authorization"] == "Bearer other"

    def test_surrounding_whitespace_is_stripped(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, page())
        OxylabsWebAPIReader(api_key=" from-file\n").load_data(["https://example.com"])
        assert web_api.requests[-1].headers["authorization"] == "Bearer from-file"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OXYLABS_WEB_API_KEY")
        with pytest.raises(ValueError, match="OXYLABS_WEB_API_KEY"):
            OxylabsWebAPIReader()

    def test_whitespace_inside_is_rejected_without_quoting_it(self) -> None:
        with pytest.raises(ValueError, match="whitespace") as error:
            OxylabsWebAPIReader(api_key="secret\nkey")
        assert "secret" not in str(error.value)

    def test_is_never_serialized(self) -> None:
        reader = OxylabsWebAPIReader(api_key="secret-key")
        assert "secret-key" not in repr(reader)
        assert "secret-key" not in reader.model_dump_json()


class TestLoadData:
    def test_one_document_per_url(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, page("first", "https://a.example/")).add(200, page("second"))
        docs = OxylabsWebAPIReader().load_data(
            ["https://a.example", "https://b.example"]
        )
        assert [d.text for d in docs] == ["first", "second"]
        assert docs[0].metadata == {
            "source": "https://a.example",
            "final_url": "https://a.example/",
            "status_code": 200,
            "request_id": "42",
        }

    def test_sends_markdown_by_default(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, page())
        OxylabsWebAPIReader().load_data(["https://example.com"])
        assert web_api.payloads == [
            {"url": "https://example.com", "output": ["markdown"]}
        ]

    def test_sends_the_reader_options(self, web_api: FakeWebAPI) -> None:
        web_api.add(
            200, {"state": "done", "results": [{"html": "<p>x</p>", "metadata": {}}]}
        )
        docs = OxylabsWebAPIReader(
            output="html", location="de", device="mobile", run_js=True
        ).load_data(["https://example.com"])
        assert docs[0].text == "<p>x</p>"
        assert web_api.payloads == [
            {
                "url": "https://example.com",
                "output": ["html"],
                "location": "DE",
                "device": "mobile",
                "run_js": True,
            }
        ]

    def test_place_name_location_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            OxylabsWebAPIReader(location="Germany")

    def test_identifies_the_integration(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, page())
        OxylabsWebAPIReader().load_data(["https://example.com"])
        sdk = web_api.requests[-1].headers["x-oxylabs-sdk"]
        assert sdk.startswith("oxylabs-llama-index-web-sdk-python/")

    def test_slow_page_is_queued_and_collected(self, web_api: FakeWebAPI) -> None:
        web_api.add(504, {"title": "REQUEST_FAILED_SCRAPE_TIMEOUT"})
        web_api.add(202, {"state": "pending", "metadata": {"request_id": "7"}})
        web_api.add(200, {"state": "pending", "metadata": {"request_id": "7"}})
        web_api.add(200, page())
        docs = OxylabsWebAPIReader().load_data(["https://example.com"])
        assert docs[0].text == PAGE
        assert web_api.paths == [
            "POST /v1/scrape",
            "POST /v1/async/scrape",
            "GET /v1/async/scrape/7",
            "GET /v1/async/scrape/7",
        ]

    def test_transient_error_is_retried(self, web_api: FakeWebAPI) -> None:
        web_api.add(503, {"detail": "busy"}).add(200, page())
        assert len(OxylabsWebAPIReader().load_data(["https://example.com"])) == 1
        assert len(web_api.requests) == 2

    def test_connection_error_is_retried(self, web_api: FakeWebAPI) -> None:
        web_api.add(0, httpx.ConnectError("down")).add(200, page())
        assert len(OxylabsWebAPIReader().load_data(["https://example.com"])) == 1


class TestFailures:
    def test_failed_url_is_skipped_with_a_warning(self, web_api: FakeWebAPI) -> None:
        web_api.add(500, FAULTED).add(200, page("second"))
        with pytest.warns(UserWarning, match="https://a.example.*request_id 99"):
            docs = OxylabsWebAPIReader().load_data(
                ["https://a.example", "https://b.example"]
            )
        assert [d.text for d in docs] == ["second"]

    def test_faulted_request_is_not_resent(self, web_api: FakeWebAPI) -> None:
        web_api.add(500, FAULTED)
        with pytest.warns(UserWarning):
            OxylabsWebAPIReader().load_data(["https://example.com"])
        assert len(web_api.requests) == 1

    def test_page_that_never_finishes_is_skipped(self, web_api: FakeWebAPI) -> None:
        web_api.add(504, {"title": "REQUEST_FAILED_SCRAPE_TIMEOUT"})
        web_api.add(200, {"state": "pending", "metadata": {"request_id": "7"}})
        with pytest.warns(UserWarning, match="max_wait"):
            docs = OxylabsWebAPIReader(max_wait=0).load_data(["https://example.com"])
        assert docs == []

    def test_bad_key_raises_instead_of_skipping(self, web_api: FakeWebAPI) -> None:
        web_api.add(401, {})
        with pytest.raises(OxylabsWebAPIError, match="OXYLABS_WEB_API_KEY"):
            OxylabsWebAPIReader().load_data(["https://a.example", "https://b.example"])
        assert len(web_api.requests) == 1

    def test_spent_quota_raises(self, web_api: FakeWebAPI) -> None:
        web_api.add(429, {"title": "QUOTA_EXCEEDED"})
        with pytest.raises(OxylabsWebAPIError, match="quota"):
            OxylabsWebAPIReader().load_data(["https://example.com"])
        assert len(web_api.requests) == 1

    def test_rejected_header_never_echoes_the_key(self, web_api: FakeWebAPI) -> None:
        web_api.add(0, httpx.LocalProtocolError("Illegal header value b'Bearer k'"))
        with pytest.raises(OxylabsWebAPIError, match="headers were rejected") as error:
            OxylabsWebAPIReader().load_data(["https://example.com"])
        assert "Bearer" not in str(error.value)


class TestAsync:
    @pytest.mark.asyncio
    async def test_loads_every_url(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, page())
        docs = await OxylabsWebAPIReader().aload_data(
            ["https://a.example", "https://b.example", "https://c.example"]
        )
        assert len(docs) == 3
        assert {d.metadata["source"] for d in docs} == {
            "https://a.example",
            "https://b.example",
            "https://c.example",
        }

    @pytest.mark.asyncio
    async def test_failed_url_is_skipped(self, web_api: FakeWebAPI) -> None:
        web_api.add(500, FAULTED)
        with pytest.warns(UserWarning, match="request_id 99"):
            docs = await OxylabsWebAPIReader().aload_data(["https://example.com"])
        assert docs == []

    @pytest.mark.asyncio
    async def test_slow_page_is_queued_and_collected(self, web_api: FakeWebAPI) -> None:
        web_api.add(504, {"title": "REQUEST_FAILED_SCRAPE_TIMEOUT"})
        web_api.add(202, {"state": "pending", "metadata": {"request_id": "7"}})
        web_api.add(200, page())
        docs = await OxylabsWebAPIReader(poll_interval=0).aload_data(
            ["https://example.com"]
        )
        assert docs[0].text == PAGE
        assert web_api.paths[-1] == "GET /v1/async/scrape/7"

    @pytest.mark.asyncio
    async def test_bad_key_raises(self, web_api: FakeWebAPI) -> None:
        web_api.add(401, {})
        with pytest.raises(OxylabsWebAPIError):
            await OxylabsWebAPIReader().aload_data(["https://example.com"])
