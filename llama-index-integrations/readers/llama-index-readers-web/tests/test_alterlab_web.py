"""Tests for AlterLabWebReader."""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.schema import Document
from llama_index.readers.web import AlterLabWebReader

MODULE = "llama_index.readers.web.alterlab_web.base"


def _response(data, status=200):
    response = MagicMock()
    response.status_code = status
    response.ok = status < 400
    response.text = str(data)
    response.json.return_value = data
    return response


@pytest.fixture(autouse=True)
def no_sleep():
    with patch(f"{MODULE}.time.sleep"):
        yield


def test_scrape_returns_markdown_document():
    payload = {
        "content": {"markdown": "# Hello", "text": "Hello"},
        "metadata": {"title": "Example", "status_code": 200},
    }
    with patch(f"{MODULE}.requests.request", return_value=_response(payload)) as req:
        docs = AlterLabWebReader(api_key="sk_test").load_data(url="https://example.com")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "# Hello"
    assert docs[0].metadata == {
        "source": "https://example.com",
        "title": "Example",
        "status_code": 200,
    }
    args, kwargs = req.call_args
    assert args == ("POST", "https://api.alterlab.io/api/v1/scrape")
    assert kwargs["headers"] == {"X-API-Key": "sk_test"}
    assert kwargs["json"]["url"] == "https://example.com"
    assert kwargs["json"]["sync"] is True


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("ALTERLAB_API_KEY", "sk_env")
    with patch(
        f"{MODULE}.requests.request",
        return_value=_response({"content": "body"}),
    ) as req:
        AlterLabWebReader().load_data(url="https://example.com")

    assert req.call_args.kwargs["headers"] == {"X-API-Key": "sk_env"}


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ALTERLAB_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ALTERLAB_API_KEY"):
        AlterLabWebReader().load_data(url="https://example.com")


def test_params_are_merged_into_request():
    reader = AlterLabWebReader(api_key="sk_test", params={"mode": "js"})
    with patch(
        f"{MODULE}.requests.request", return_value=_response({"content": "x"})
    ) as req:
        reader.load_data(url="https://example.com", params={"cache": "skip"})

    body = req.call_args.kwargs["json"]
    assert body["mode"] == "js"
    assert body["cache"] == "skip"


def test_scrape_polls_job_when_accepted():
    responses = [
        _response({"job_id": "job_1"}, status=202),
        _response({"status": "running"}),
        _response({"status": "succeeded", "result": {"content": {"markdown": "done"}}}),
    ]
    with patch(f"{MODULE}.requests.request", side_effect=responses) as req:
        docs = AlterLabWebReader(api_key="sk_test").load_data(url="https://example.com")

    assert docs[0].text == "done"
    assert req.call_args_list[1].args == (
        "GET",
        "https://api.alterlab.io/api/v1/jobs/job_1",
    )


def test_failed_job_raises():
    responses = [
        _response({"job_id": "job_1"}, status=202),
        _response({"status": "failed", "error": "blocked"}),
    ]
    with patch(f"{MODULE}.requests.request", side_effect=responses):
        with pytest.raises(RuntimeError, match="blocked"):
            AlterLabWebReader(api_key="sk_test").load_data(url="https://example.com")


def test_api_error_raises_with_status():
    with patch(
        f"{MODULE}.requests.request",
        return_value=_response({"detail": "bad key"}, status=401),
    ):
        with pytest.raises(RuntimeError, match="401"):
            AlterLabWebReader(api_key="sk_bad").load_data(url="https://example.com")


def test_crawl_returns_one_document_per_page():
    responses = [
        _response({"crawl_id": "c1"}),
        _response(
            {
                "status": "completed",
                "pages": [
                    {"url": "https://a.test", "content": {"markdown": "A"}},
                    {"url": "https://a.test/b", "content": {"markdown": "B"}},
                    {"url": "https://a.test/c", "content": {}},
                ],
            }
        ),
    ]
    reader = AlterLabWebReader(api_key="sk_test", mode="crawl", params={"max_pages": 3})
    with patch(f"{MODULE}.requests.request", side_effect=responses) as req:
        docs = reader.load_data(url="https://a.test")

    assert [d.text for d in docs] == ["A", "B"]
    assert [d.metadata["source"] for d in docs] == [
        "https://a.test",
        "https://a.test/b",
    ]
    assert req.call_args_list[0].kwargs["json"]["max_pages"] == 3
    assert req.call_args_list[1].kwargs["params"] == {"include_results": "true"}


def test_search_returns_one_document_per_result():
    payload = {
        "results": [
            {
                "title": "One",
                "url": "https://one.test",
                "snippet": "first",
                "position": 1,
            },
            {"title": "Two", "url": "https://two.test"},
        ]
    }
    reader = AlterLabWebReader(api_key="sk_test", mode="search")
    with patch(f"{MODULE}.requests.request", return_value=_response(payload)):
        docs = reader.load_data(query="scraping")

    assert [d.text for d in docs] == ["One\nfirst", "Two"]
    assert docs[0].metadata == {
        "source": "https://one.test",
        "title": "One",
        "query": "scraping",
        "position": 1,
    }


def test_invalid_mode_and_missing_inputs():
    with pytest.raises(ValueError, match="Invalid mode"):
        AlterLabWebReader(api_key="k", mode="nope").load_data(url="https://a.test")
    with pytest.raises(ValueError, match="url"):
        AlterLabWebReader(api_key="k").load_data()
    with pytest.raises(ValueError, match="query"):
        AlterLabWebReader(api_key="k", mode="search").load_data()


def test_class_name_and_api_key_not_serialized():
    reader = AlterLabWebReader(api_key="sk_secret")

    assert AlterLabWebReader.class_name() == "AlterLabWebReader"
    assert "sk_secret" not in reader.to_json()
