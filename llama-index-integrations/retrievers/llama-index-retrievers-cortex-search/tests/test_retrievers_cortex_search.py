"""Tests for llama-index-retrievers-cortex-search."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.core.schema import QueryBundle
from llama_index.retrievers.cortex_search.base import CortexSearchRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_search_response(results: list) -> dict:
    """Build a Cortex Search API response."""
    return {
        "results": results,
        "request_id": "test-req-id-123",
    }


def _make_retriever(**kwargs) -> CortexSearchRetriever:
    """Create a CortexSearchRetriever with test JWT (no real auth)."""
    defaults = {
        "service_name": "my_svc",
        "database": "MY_DB",
        "schema": "MY_SCHEMA",
        "search_column": "content",
        "columns": ["content", "title", "url"],
        "jwt_token": "test-jwt",
        "account": "ORG-ACCT",
        "user": "TESTUSER",
    }
    defaults.update(kwargs)
    return CortexSearchRetriever(**defaults)


# ---------------------------------------------------------------------------
# Construction / auth tests
# ---------------------------------------------------------------------------


class TestCortexSearchRetrieverInit:
    def test_init_with_jwt(self) -> None:
        r = _make_retriever()
        assert r.service_name == "my_svc"
        assert r.database == "MY_DB"
        assert r.schema == "MY_SCHEMA"
        assert r.jwt_token == "test-jwt"
        assert r.limit == 10

    def test_init_custom_limit(self) -> None:
        r = _make_retriever(limit=5)
        assert r.limit == 5

    def test_search_column_auto_included(self) -> None:
        r = _make_retriever(
            search_column="body",
            columns=["title", "url"],
        )
        assert "body" in r.columns
        assert r.columns[0] == "body"

    def test_init_with_env_key_file(self, tmp_path) -> None:
        key_path = str(tmp_path / "rsa_key.p8")
        with open(key_path, "w") as f:
            f.write("placeholder")
        with (
            patch.dict(
                "os.environ",
                {
                    "SNOWFLAKE_KEY_FILE": key_path,
                    "SNOWFLAKE_USERNAME": "USR",
                    "SNOWFLAKE_ACCOUNT": "ORG-ACCT",
                },
            ),
            patch(
                "llama_index.retrievers.cortex_search.base.is_spcs_environment",
                return_value=False,
            ),
        ):
            r = CortexSearchRetriever(
                service_name="svc",
                database="DB",
                schema="SCH",
                search_column="content",
            )
            assert r.private_key_file == key_path

    def test_init_no_auth_raises(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "llama_index.retrievers.cortex_search.base.is_spcs_environment",
                return_value=False,
            ),
        ):
            with pytest.raises(ValueError, match="Authentication required"):
                CortexSearchRetriever(
                    service_name="svc",
                    database="DB",
                    schema="SCH",
                    search_column="content",
                    account="ORG-ACCT",
                    user="USR",
                )


# ---------------------------------------------------------------------------
# Auth header tests
# ---------------------------------------------------------------------------


class TestAuthHeader:
    def test_jwt_auth_header(self) -> None:
        r = _make_retriever(jwt_token="my-jwt-456")
        assert r._generate_auth_header() == "Bearer my-jwt-456"

    def test_session_auth_header(self) -> None:
        mock_session = MagicMock()
        mock_session.connection.rest.token = "sess-tok-789"
        r = _make_retriever(jwt_token=None, session=mock_session)
        assert r._generate_auth_header() == 'Snowflake Token="sess-tok-789"'


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_search_endpoint(self) -> None:
        r = _make_retriever()
        with patch(
            "llama_index.retrievers.cortex_search.base.is_spcs_environment",
            return_value=False,
        ):
            expected = (
                "https://ORG-ACCT.snowflakecomputing.com"
                "/api/v2/databases/MY_DB"
                "/schemas/MY_SCHEMA"
                "/cortex-search-services/my_svc:query"
            )
            assert r._search_endpoint == expected


# ---------------------------------------------------------------------------
# Request body tests
# ---------------------------------------------------------------------------


class TestRequestBody:
    def test_basic_body(self) -> None:
        r = _make_retriever()
        body = r._build_request_body("test query")
        assert body["query"] == "test query"
        assert body["columns"] == ["content", "title", "url"]
        assert body["limit"] == 10
        assert "filter" not in body

    def test_body_with_filter(self) -> None:
        r = _make_retriever(filter_spec={"@eq": {"category": "docs"}})
        body = r._build_request_body("test")
        assert body["filter"] == {"@eq": {"category": "docs"}}


# ---------------------------------------------------------------------------
# Sync retrieval tests
# ---------------------------------------------------------------------------


class TestSyncRetrieval:
    @patch("llama_index.retrievers.cortex_search.base.requests.post")
    def test_retrieve_basic(self, mock_post) -> None:
        results = [
            {"content": "Result one", "title": "Title 1", "url": "/a"},
            {"content": "Result two", "title": "Title 2", "url": "/b"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_search_response(results)
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        r = _make_retriever()
        with patch(
            "llama_index.retrievers.cortex_search.base.is_spcs_environment",
            return_value=False,
        ):
            nodes = r._retrieve(QueryBundle(query_str="test query"))

        assert len(nodes) == 2
        assert nodes[0].node.text == "Result one"
        assert nodes[0].node.metadata["title"] == "Title 1"
        assert nodes[1].node.text == "Result two"

        # Verify the API call
        call_kwargs = mock_post.call_args
        assert "test query" in call_kwargs.kwargs["json"]["query"]

    @patch("llama_index.retrievers.cortex_search.base.requests.post")
    def test_retrieve_with_score(self, mock_post) -> None:
        results = [
            {
                "content": "Scored result",
                "title": "T",
                "@search_score": 0.95,
            },
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_search_response(results)
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        r = _make_retriever()
        with patch(
            "llama_index.retrievers.cortex_search.base.is_spcs_environment",
            return_value=False,
        ):
            nodes = r._retrieve(QueryBundle(query_str="scored"))

        assert nodes[0].score == 0.95

    @patch("llama_index.retrievers.cortex_search.base.requests.post")
    def test_retrieve_empty_results(self, mock_post) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_search_response([])
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        r = _make_retriever()
        with patch(
            "llama_index.retrievers.cortex_search.base.is_spcs_environment",
            return_value=False,
        ):
            nodes = r._retrieve(QueryBundle(query_str="nothing"))

        assert len(nodes) == 0


# ---------------------------------------------------------------------------
# Async retrieval tests
# ---------------------------------------------------------------------------


class TestAsyncRetrieval:
    @pytest.mark.asyncio
    async def test_aretrieve(self) -> None:
        results = [
            {"content": "Async result", "title": "AT", "url": "/c"},
        ]
        resp_data = _mock_search_response(results)

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=resp_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        r = _make_retriever()
        with (
            patch(
                "llama_index.retrievers.cortex_search.base.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch(
                "llama_index.retrievers.cortex_search.base.is_spcs_environment",
                return_value=False,
            ),
        ):
            nodes = await r._aretrieve(QueryBundle(query_str="async query"))

        assert len(nodes) == 1
        assert nodes[0].node.text == "Async result"
