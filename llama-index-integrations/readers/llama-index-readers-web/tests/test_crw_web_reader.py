from unittest.mock import MagicMock, patch

import pytest

from llama_index.readers.web.crw_web.base import CrwReader, CrwWebReader


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    if status_code >= 400:
        from requests import HTTPError

        mock.raise_for_status.side_effect = HTTPError(response=mock)
    else:
        mock.raise_for_status.return_value = None
    return mock


def test_class_name():
    assert CrwWebReader.class_name() == "CrwWeb_reader"


def test_is_remote():
    reader = CrwWebReader()
    assert reader.is_remote is True


def test_invalid_mode_raises_value_error():
    with pytest.raises(ValueError, match="Invalid mode"):
        CrwWebReader(mode="invalid")


def test_api_key_sets_auth_header():
    reader = CrwWebReader(api_key="secret")
    assert reader._session.headers.get("Authorization") == "Bearer secret"


def test_no_api_key_sets_no_auth_header():
    reader = CrwWebReader()
    assert "Authorization" not in reader._session.headers


def test_empty_url_raises():
    reader = CrwWebReader()
    with pytest.raises(ValueError, match="url must not be empty"):
        reader.load_data(url="")


def test_load_data_invalid_mode_kwarg_raises():
    reader = CrwWebReader(mode="scrape")
    with pytest.raises(ValueError, match="Invalid mode"):
        reader.load_data(url="https://site", mode="invalid")


def test_load_data_mode_overrides_reader_mode():
    reader = CrwWebReader(mode="scrape")
    payload = {
        "success": True,
        "links": [{"url": "https://site/a", "title": "A"}],
    }
    with patch.object(reader._session, "post", return_value=_mock_response(payload)):
        docs = reader.load_data(url="https://site", mode="map")

    assert len(docs) == 1
    assert docs[0].metadata.get("source") == "map"
    assert docs[0].metadata["source_url"] == "https://site/a"


def test_map_uses_sourceurl_when_page_url_missing():
    reader = CrwWebReader(mode="scrape")
    payload = {
        "success": True,
        "data": {
            "markdown": "x",
            "metadata": {"sourceURL": "https://from-meta"},
        },
    }
    with patch.object(reader._session, "post", return_value=_mock_response(payload)):
        docs = reader.load_data(url="https://site")

    assert docs[0].metadata["source_url"] == "https://from-meta"


def test_scrape_returns_document_with_text_and_metadata():
    reader = CrwWebReader(mode="scrape")
    payload = {
        "success": True,
        "data": {
            "markdown": "# Hello",
            "url": "https://site",
            "title": "Example",
            "statusCode": 200,
            "metadata": {"lang": "en"},
        },
    }
    with patch.object(reader._session, "post", return_value=_mock_response(payload)):
        docs = reader.load_data(url="https://site")

    assert len(docs) == 1
    assert docs[0].text == "# Hello"
    assert docs[0].metadata["source_url"] == "https://site"
    assert docs[0].metadata["title"] == "Example"
    assert docs[0].metadata["statusCode"] == 200
    assert docs[0].metadata["lang"] == "en"


def test_scrape_always_sends_formats_markdown():
    reader = CrwWebReader(mode="scrape")
    payload = {"success": True, "data": {"markdown": "text", "metadata": {}}}
    with patch.object(
        reader._session, "post", return_value=_mock_response(payload)
    ) as mock_post:
        reader.load_data(url="https://site")

    called_body = mock_post.call_args[1]["json"]
    assert called_body.get("formats") == ["markdown"]


def test_scrape_api_error_raises_runtime_error():
    reader = CrwWebReader(mode="scrape")
    payload = {"success": False, "error": "rate limit exceeded"}
    with patch.object(reader._session, "post", return_value=_mock_response(payload)):
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            reader.load_data(url="https://site")


def test_scrape_http_error_raises():
    reader = CrwWebReader(mode="scrape")
    with patch.object(
        reader._session, "post", return_value=_mock_response({}, status_code=500)
    ):
        with pytest.raises(Exception):
            reader.load_data(url="https://site")


def test_crawl_polls_until_completed_and_returns_docs():
    reader = CrwWebReader(mode="crawl", poll_interval=0)

    submit_resp = _mock_response({"success": True, "id": "job-123"})
    pending_resp = _mock_response({"status": "running"})
    done_resp = _mock_response(
        {
            "status": "completed",
            "data": [
                {
                    "markdown": "Page 1",
                    "url": "https://site/1",
                    "title": "P1",
                    "statusCode": 200,
                    "metadata": {},
                },
                {
                    "markdown": "Page 2",
                    "url": "https://site/2",
                    "title": "P2",
                    "statusCode": 200,
                    "metadata": {},
                },
            ],
        }
    )

    with patch.object(reader._session, "post", return_value=submit_resp):
        with patch.object(
            reader._session, "get", side_effect=[pending_resp, done_resp]
        ):
            docs = reader.load_data(url="https://site")

    assert len(docs) == 2
    assert docs[0].text == "Page 1"
    assert docs[1].text == "Page 2"
    assert docs[0].metadata["source_url"] == "https://site/1"


def test_crawl_bad_status_raises_with_status_in_message():
    reader = CrwWebReader(mode="crawl", poll_interval=0)

    submit_resp = _mock_response({"success": True, "id": "job-456"})
    failed_resp = _mock_response({"status": "failed"})

    with patch.object(reader._session, "post", return_value=submit_resp):
        with patch.object(reader._session, "get", return_value=failed_resp):
            with pytest.raises(RuntimeError, match="failed"):
                reader.load_data(url="https://site")


def test_crawl_timeout_raises():
    reader = CrwWebReader(mode="crawl", poll_interval=0, poll_timeout=0)

    submit_resp = _mock_response({"success": True, "id": "job-789"})
    running_resp = _mock_response({"status": "running"})

    with patch.object(reader._session, "post", return_value=submit_resp):
        with patch.object(reader._session, "get", return_value=running_resp):
            with pytest.raises(RuntimeError, match="did not complete"):
                reader.load_data(url="https://site")


def test_crawl_missing_job_id_raises():
    reader = CrwWebReader(mode="crawl", poll_interval=0)
    submit_resp = _mock_response({"success": True})  # no id field

    with patch.object(reader._session, "post", return_value=submit_resp):
        with pytest.raises(RuntimeError, match="job id"):
            reader.load_data(url="https://site")


def test_map_returns_one_doc_per_link():
    reader = CrwWebReader(mode="map")
    payload = {
        "success": True,
        "links": [
            {"url": "https://site/a", "title": "A"},
            {"url": "https://site/b", "title": "B"},
        ],
    }
    with patch.object(reader._session, "post", return_value=_mock_response(payload)):
        docs = reader.load_data(url="https://site")

    assert len(docs) == 2
    assert docs[0].metadata["source_url"] == "https://site/a"
    assert docs[0].metadata["source"] == "map"
    assert docs[1].metadata["title"] == "B"


def test_map_handles_string_links():
    reader = CrwWebReader(mode="map")
    payload = {"success": True, "links": ["https://site/a", "https://site/b"]}
    with patch.object(reader._session, "post", return_value=_mock_response(payload)):
        docs = reader.load_data(url="https://site")

    assert len(docs) == 2
    assert docs[0].metadata["source_url"] == "https://site/a"


def test_map_api_error_raises():
    reader = CrwWebReader(mode="map")
    payload = {"success": False, "error": "not found"}
    with patch.object(reader._session, "post", return_value=_mock_response(payload)):
        with pytest.raises(RuntimeError, match="not found"):
            reader.load_data(url="https://site")


def test_crw_reader_class_name():
    assert CrwReader.class_name() == "CrwReader"


def test_crw_reader_load_data_with_mode_kwarg():
    reader = CrwReader(mode="scrape")
    payload = {
        "success": True,
        "data": {
            "markdown": "# Hi",
            "url": "https://site",
            "title": "T",
            "metadata": {},
        },
    }
    with patch.object(reader._session, "post", return_value=_mock_response(payload)):
        docs = reader.load_data(url="https://site", mode="scrape")

    assert len(docs) == 1
    assert docs[0].text == "# Hi"
