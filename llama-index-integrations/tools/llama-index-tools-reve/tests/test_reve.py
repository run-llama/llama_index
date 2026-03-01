"""Tests for Reve AI tool spec."""

import base64
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.reve.base import (
    ReveToolSpec,
)


FAKE_API_KEY = "test-api-key-123"

MOCK_RESPONSE_HEADERS = {
    "x-credits-used": "2",
    "x-credits-remaining": "498",
    "x-model-version": "reve-v1",
    "x-request-id": "req-abc123",
    "x-content-violation": "false",
}

MOCK_CREATE_BODY = {
    "data": [{"url": "https://cdn.reve.com/images/generated-123.png"}]
}

MOCK_IMAGE_BYTES = b"\x89PNG\r\n\x1a\nfake-image-data"


def _mock_response(
    status_code: int = 200,
    json_body: dict = None,
    headers: dict = None,
    content: bytes = None,
) -> httpx.Response:
    """Build a mock httpx.Response."""
    actual_headers = MOCK_RESPONSE_HEADERS if headers is None else headers
    kwargs: dict = {
        "status_code": status_code,
        "headers": actual_headers,
        "request": httpx.Request("POST", "https://api.reve.com/test"),
    }
    if json_body is not None:
        kwargs["json"] = json_body
    elif content is not None:
        kwargs["content"] = content
    else:
        kwargs["content"] = b""
    return httpx.Response(**kwargs)


class TestReveToolSpecInit:
    def test_inherits_base_tool_spec(self) -> None:
        assert issubclass(ReveToolSpec, BaseToolSpec)

    def test_spec_functions_defined(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY)
        assert "reve_create_image" in spec.spec_functions
        assert "reve_edit_image" in spec.spec_functions
        assert "reve_remix_image" in spec.spec_functions
        assert "reve_check_credits" in spec.spec_functions
        assert len(spec.spec_functions) == 4

    def test_api_key_from_param(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY)
        assert spec.api_key == FAKE_API_KEY

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REVE_API_KEY", "env-key-456")
        spec = ReveToolSpec()
        assert spec.api_key == "env-key-456"

    def test_api_key_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REVE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Reve API key is required"):
            ReveToolSpec()

    def test_custom_base_url(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY, base_url="https://custom.reve.com/")
        assert spec.base_url == "https://custom.reve.com"


class TestParseResponseHeaders:
    def test_parses_all_known_headers(self) -> None:
        resp = _mock_response(headers=MOCK_RESPONSE_HEADERS)
        meta = ReveToolSpec._parse_response_headers(resp)
        assert meta["credits_used"] == "2"
        assert meta["credits_remaining"] == "498"
        assert meta["model_version"] == "reve-v1"
        assert meta["request_id"] == "req-abc123"
        assert meta["content_violation"] == "false"

    def test_handles_missing_headers(self) -> None:
        resp = _mock_response(headers={})
        meta = ReveToolSpec._parse_response_headers(resp)
        assert meta == {}

    def test_handles_partial_headers(self) -> None:
        resp = _mock_response(headers={"x-credits-remaining": "100"})
        meta = ReveToolSpec._parse_response_headers(resp)
        assert meta == {"credits_remaining": "100"}


class TestBuildResult:
    def test_combines_body_and_headers(self) -> None:
        resp = _mock_response(headers=MOCK_RESPONSE_HEADERS)
        result = ReveToolSpec._build_result(resp, {"data": [{"url": "test"}]})
        parsed = json.loads(result)
        assert parsed["data"] == [{"url": "test"}]
        assert "meta" in parsed
        assert parsed["meta"]["credits_used"] == "2"


@pytest.mark.asyncio
class TestReveCreateImage:
    async def test_basic_create(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY)
        mock_resp = _mock_response(json_body=MOCK_CREATE_BODY)

        with patch.object(
            spec._get_client().__class__,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await spec.reve_create_image(prompt="A sunset")

        parsed = json.loads(result)
        assert "data" in parsed
        assert parsed["data"][0]["url"].startswith("https://")
        assert "meta" in parsed

    async def test_create_with_all_params(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY)
        mock_resp = _mock_response(json_body=MOCK_CREATE_BODY)

        with patch.object(
            spec._get_client().__class__,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await spec.reve_create_image(
                prompt="A sunset",
                aspect_ratio="16:9",
                test_time_scaling=5,
                upscale_factor=2,
                remove_background=True,
                fit_max_dim=1024,
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert payload["prompt"] == "A sunset"
            assert payload["aspect_ratio"] == "16:9"
            assert payload["test_time_scaling"] == 5
            assert payload["upscale_factor"] == 2
            assert payload["remove_background"] is True
            assert payload["fit_max_dim"] == 1024

    async def test_create_omits_none_params(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY)
        mock_resp = _mock_response(json_body=MOCK_CREATE_BODY)

        with patch.object(
            spec._get_client().__class__,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await spec.reve_create_image(prompt="test")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "test_time_scaling" not in payload
            assert "upscale_factor" not in payload
            assert "remove_background" not in payload
            assert "fit_max_dim" not in payload


@pytest.mark.asyncio
class TestReveEditImage:
    async def test_edit_fetches_and_encodes_image(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY)
        mock_image_resp = _mock_response(content=MOCK_IMAGE_BYTES, headers={})
        mock_edit_resp = _mock_response(json_body=MOCK_CREATE_BODY)

        async def mock_request(method_or_url, url=None, **kwargs):
            """Handle both .get() and .post() calls."""
            # If called as client.get(url) or client.post(url, ...)
            actual_url = url if url is not None else method_or_url
            if isinstance(actual_url, str) and "v1/image/edit" in actual_url:
                return mock_edit_resp
            return mock_image_resp

        client = spec._get_client()
        with (
            patch.object(
                client.__class__, "get", new_callable=AsyncMock, return_value=mock_image_resp
            ),
            patch.object(
                client.__class__, "post", new_callable=AsyncMock, return_value=mock_edit_resp
            ) as mock_post,
        ):
            result = await spec.reve_edit_image(
                prompt="Make it blue",
                image_url="https://example.com/photo.png",
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            expected_b64 = base64.b64encode(MOCK_IMAGE_BYTES).decode("utf-8")
            assert payload["image"] == expected_b64
            assert payload["prompt"] == "Make it blue"

        parsed = json.loads(result)
        assert "data" in parsed


@pytest.mark.asyncio
class TestReveRemixImage:
    async def test_remix_fetches_and_encodes_image(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY)
        mock_image_resp = _mock_response(content=MOCK_IMAGE_BYTES, headers={})
        mock_remix_resp = _mock_response(json_body=MOCK_CREATE_BODY)

        client = spec._get_client()
        with (
            patch.object(
                client.__class__, "get", new_callable=AsyncMock, return_value=mock_image_resp
            ),
            patch.object(
                client.__class__, "post", new_callable=AsyncMock, return_value=mock_remix_resp
            ) as mock_post,
        ):
            result = await spec.reve_remix_image(
                prompt="Cyberpunk style",
                image_url="https://example.com/photo.png",
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            expected_b64 = base64.b64encode(MOCK_IMAGE_BYTES).decode("utf-8")
            assert payload["image"] == expected_b64

        parsed = json.loads(result)
        assert "data" in parsed


@pytest.mark.asyncio
class TestReveCheckCredits:
    async def test_returns_credit_info(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY)
        mock_resp = _mock_response(json_body=MOCK_CREATE_BODY)

        with patch.object(
            spec._get_client().__class__,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await spec.reve_check_credits()

        parsed = json.loads(result)
        assert parsed["credits_remaining"] == "498"
        assert parsed["credits_used"] == "2"

    async def test_handles_missing_credit_headers(self) -> None:
        spec = ReveToolSpec(api_key=FAKE_API_KEY)
        mock_resp = _mock_response(json_body=MOCK_CREATE_BODY, headers={})

        with patch.object(
            spec._get_client().__class__,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await spec.reve_check_credits()

        parsed = json.loads(result)
        assert parsed["credits_remaining"] == "unknown"
        assert parsed["credits_used"] == "unknown"
