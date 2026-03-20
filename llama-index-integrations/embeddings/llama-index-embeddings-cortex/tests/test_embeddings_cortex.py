"""Tests for llama-index-embeddings-cortex."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.embeddings.cortex.base import (
    CortexEmbedding,
    DEFAULT_MODEL,
    EMBED_MODELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_embed_response(embeddings: list) -> dict:
    """Build a Cortex embed API response dict."""
    return {
        "data": [
            {"embedding": [emb], "index": i, "object": "embedding"}
            for i, emb in enumerate(embeddings)
        ],
        "model": DEFAULT_MODEL,
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5,
        },
    }


def _make_embedding(jwt_token: str = "test-jwt") -> CortexEmbedding:
    """Create a CortexEmbedding with a test JWT (no real auth)."""
    return CortexEmbedding.model_construct(
        model_name=DEFAULT_MODEL,
        jwt_token=jwt_token,
        account="ORG-ACCT",
        user="TESTUSER",
        embed_batch_size=32,
    )


# ---------------------------------------------------------------------------
# Construction / auth tests
# ---------------------------------------------------------------------------


class TestCortexEmbeddingInit:
    def test_init_with_jwt(self) -> None:
        emb = _make_embedding()
        assert emb.model_name == DEFAULT_MODEL
        assert emb.jwt_token == "test-jwt"

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
                "llama_index.embeddings.cortex.base.is_spcs_environment",
                return_value=False,
            ),
        ):
            emb = CortexEmbedding()
            assert emb.private_key_file == key_path

    def test_init_no_auth_raises(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "llama_index.embeddings.cortex.base.is_spcs_environment",
                return_value=False,
            ),
        ):
            with pytest.raises(ValueError, match="Authentication required"):
                CortexEmbedding(
                    account="ORG-ACCT",
                    user="USR",
                )


# ---------------------------------------------------------------------------
# Auth header tests
# ---------------------------------------------------------------------------


class TestAuthHeader:
    def test_jwt_auth_header(self) -> None:
        emb = _make_embedding(jwt_token="my-jwt-123")
        assert emb._generate_auth_header() == "Bearer my-jwt-123"

    def test_session_auth_header(self) -> None:
        mock_session = MagicMock()
        mock_session.connection.rest.token = "sess-tok-456"
        emb = CortexEmbedding.model_construct(
            model_name=DEFAULT_MODEL,
            session=mock_session,
            account="ORG-ACCT",
            user="USR",
            embed_batch_size=32,
        )
        assert emb._generate_auth_header() == 'Snowflake Token="sess-tok-456"'


# ---------------------------------------------------------------------------
# Endpoint / URL tests
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_embed_endpoint(self) -> None:
        emb = _make_embedding()
        with patch(
            "llama_index.embeddings.cortex.base.is_spcs_environment",
            return_value=False,
        ):
            assert emb._embed_endpoint == (
                "https://ORG-ACCT.snowflakecomputing.com/api/v2/cortex/inference:embed"
            )


# ---------------------------------------------------------------------------
# Sync embedding tests
# ---------------------------------------------------------------------------


class TestSyncEmbedding:
    @patch("llama_index.embeddings.cortex.base.requests.post")
    def test_get_text_embedding(self, mock_post) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_embed_response([vec])
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        emb = _make_embedding()
        with patch(
            "llama_index.embeddings.cortex.base.is_spcs_environment",
            return_value=False,
        ):
            result = emb._get_text_embedding("hello")

        assert result == vec
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"]["model"] == DEFAULT_MODEL
        assert call_kwargs.kwargs["json"]["text"] == ["hello"]

    @patch("llama_index.embeddings.cortex.base.requests.post")
    def test_get_query_embedding(self, mock_post) -> None:
        vec = [0.4, 0.5, 0.6]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_embed_response([vec])
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        emb = _make_embedding()
        with patch(
            "llama_index.embeddings.cortex.base.is_spcs_environment",
            return_value=False,
        ):
            result = emb._get_query_embedding("search query")

        assert result == vec

    @patch("llama_index.embeddings.cortex.base.requests.post")
    def test_get_text_embeddings_batch(self, mock_post) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_embed_response(vecs)
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        emb = _make_embedding()
        with patch(
            "llama_index.embeddings.cortex.base.is_spcs_environment",
            return_value=False,
        ):
            result = emb._get_text_embeddings(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        # Should be a single API call for batch
        assert mock_post.call_count == 1
        assert mock_post.call_args.kwargs["json"]["text"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Async embedding tests
# ---------------------------------------------------------------------------


class TestAsyncEmbedding:
    @pytest.mark.asyncio
    async def test_aget_query_embedding(self) -> None:
        vec = [0.7, 0.8, 0.9]
        resp_data = _mock_embed_response([vec])

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=resp_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        emb = _make_embedding()
        with (
            patch(
                "llama_index.embeddings.cortex.base.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch(
                "llama_index.embeddings.cortex.base.is_spcs_environment",
                return_value=False,
            ),
        ):
            result = await emb._aget_query_embedding("async query")

        assert result == vec


# ---------------------------------------------------------------------------
# Model catalog test
# ---------------------------------------------------------------------------


class TestModelCatalog:
    def test_all_models_have_dimensions(self) -> None:
        for name, spec in EMBED_MODELS.items():
            assert "dimensions" in spec, f"{name} missing dimensions"
            assert spec["dimensions"] in (768, 1024)
