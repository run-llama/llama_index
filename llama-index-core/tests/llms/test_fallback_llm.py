"""Tests for FallbackLLM -- automatic provider failover."""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.fallback import (
    FallbackLLM,
    _is_transient_error,
)
from llama_index.core.llms.mock import MockLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_failing_mock(method_names, side_effect):
    """Return a MockLLM instance with certain methods patched to raise."""
    llm = MockLLM()
    for name in method_names:
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(llm, name, MagicMock(side_effect=side_effect))
    return llm


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    def test_timeout_is_transient(self) -> None:
        exc = TimeoutError("connection timed out")
        assert _is_transient_error(exc) is True

    def test_connection_error_is_transient(self) -> None:
        exc = ConnectionError("connection refused")
        assert _is_transient_error(exc) is True

    def test_rate_limit_is_transient(self) -> None:
        exc = Exception("Too Many Requests: rate limit exceeded")
        assert _is_transient_error(exc) is True

    def test_http_429_is_transient(self) -> None:
        class Fake429(Exception):
            @property
            def status_code(self) -> int:
                return 429

        exc = Fake429("rate limited")
        assert _is_transient_error(exc) is True

    def test_http_500_is_transient(self) -> None:
        class Fake500(Exception):
            @property
            def status_code(self) -> int:
                return 500

        exc = Fake500("internal server error")
        assert _is_transient_error(exc) is True

    def test_http_503_is_transient(self) -> None:
        class Fake503(Exception):
            @property
            def status_code(self) -> int:
                return 503

        exc = Fake503("service unavailable")
        assert _is_transient_error(exc) is True

    def test_auth_error_is_permanent(self) -> None:
        exc = Exception("invalid api key: authentication failed")
        assert _is_transient_error(exc) is False

    def test_http_401_is_permanent(self) -> None:
        class Fake401(Exception):
            @property
            def status_code(self) -> int:
                return 401

        exc = Fake401("unauthorized")
        assert _is_transient_error(exc) is False

    def test_http_403_is_permanent(self) -> None:
        class Fake403(Exception):
            @property
            def status_code(self) -> int:
                return 403

        exc = Fake403("forbidden")
        assert _is_transient_error(exc) is False

    def test_invalid_request_is_permanent(self) -> None:
        exc = Exception("bad request: invalid model")
        assert _is_transient_error(exc) is False

    def test_context_length_exceeded_is_permanent(self) -> None:
        exc = Exception("context length exceeded: max tokens is 4096")
        assert _is_transient_error(exc) is False


# ---------------------------------------------------------------------------
# FallbackLLM -- happy path
# ---------------------------------------------------------------------------


class TestFallbackLLMHappyPath:
    def test_primary_succeeds_complete(self) -> None:
        primary = MockLLM()
        fallback = FallbackLLM(llms=[primary, MockLLM()])
        response = fallback.complete("hello")
        assert isinstance(response, CompletionResponse)

    def test_primary_succeeds_chat(self) -> None:
        primary = MockLLM()
        fallback = FallbackLLM(llms=[primary, MockLLM()])
        response = fallback.chat(
            [ChatMessage(role=MessageRole.USER, content="hello")]
        )
        assert isinstance(response, ChatResponse)

    def test_primary_succeeds_stream_complete(self) -> None:
        primary = MockLLM()
        fallback = FallbackLLM(llms=[primary, MockLLM()])
        gen = fallback.stream_complete("hello")
        tokens = list(gen)
        assert isinstance(tokens, list)

    def test_primary_succeeds_stream_chat(self) -> None:
        primary = MockLLM()
        fallback = FallbackLLM(llms=[primary, MockLLM()])
        gen = fallback.stream_chat(
            [ChatMessage(role=MessageRole.USER, content="hello")]
        )
        responses = list(gen)
        assert isinstance(responses, list)

    @pytest.mark.asyncio()
    async def test_primary_succeeds_acomplete(self) -> None:
        primary = MockLLM()
        fallback = FallbackLLM(llms=[primary, MockLLM()])
        response = await fallback.acomplete("hello")
        assert isinstance(response, CompletionResponse)

    @pytest.mark.asyncio()
    async def test_primary_succeeds_achat(self) -> None:
        primary = MockLLM()
        fallback = FallbackLLM(llms=[primary, MockLLM()])
        response = await fallback.achat(
            [ChatMessage(role=MessageRole.USER, content="hello")]
        )
        assert isinstance(response, ChatResponse)

    @pytest.mark.asyncio()
    async def test_primary_succeeds_astream_complete(self) -> None:
        primary = MockLLM()
        fallback = FallbackLLM(llms=[primary, MockLLM()])
        agen = await fallback.astream_complete("hello")
        tokens = [token async for token in agen]
        assert isinstance(tokens, list)

    @pytest.mark.asyncio()
    async def test_primary_succeeds_astream_chat(self) -> None:
        primary = MockLLM()
        fallback = FallbackLLM(llms=[primary, MockLLM()])
        agen = await fallback.astream_chat(
            [ChatMessage(role=MessageRole.USER, content="hello")]
        )
        responses = [r async for r in agen]
        assert isinstance(responses, list)

    def test_metadata_reflects_primary(self) -> None:
        # Create a custom subclass to control metadata
        class CustomMetaMockLLM(MockLLM):
            @property
            def metadata(self) -> LLMMetadata:
                return LLMMetadata(
                    is_chat_model=True,
                    is_function_calling_model=True,
                    model_name="gpt-4o",
                    context_window=128000,
                    num_output=4096,
                )

        primary = CustomMetaMockLLM()
        fallback_llm = MockLLM()
        # Patch fallback's metadata using object.__setattr__ on the property
        # Actually, we need to override the metadata property on the instance
        fallback_llm.__class__ = type(
            "CustomFallbackMockLLM",
            (MockLLM,),
            {
                "metadata": property(
                    lambda self: LLMMetadata(model_name="claude-3")  # type: ignore[arg-type]
                )
            },
        )
        fallback = FallbackLLM(llms=[primary, fallback_llm])
        assert fallback.metadata.is_chat_model is True
        assert fallback.metadata.is_function_calling_model is True
        assert "gpt-4o" in fallback.metadata.model_name
        assert "claude-3" in fallback.metadata.model_name

    def test_class_name(self) -> None:
        assert FallbackLLM.class_name() == "fallback_llm"

    def test_empty_llms_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one LLM"):
            FallbackLLM(llms=[])


# ---------------------------------------------------------------------------
# FallbackLLM -- failover (sync)
# ---------------------------------------------------------------------------


class TestFallbackLLMFailoverSync:
    def test_falls_back_on_transient_complete(self) -> None:
        primary = _make_failing_mock(["complete"], TimeoutError("timed out"))
        backup = MockLLM()
        fallback = FallbackLLM(llms=[primary, backup])
        response = fallback.complete("hello")
        assert isinstance(response, CompletionResponse)

    def test_falls_back_on_transient_chat(self) -> None:
        primary = _make_failing_mock(["chat"], ConnectionError("connection refused"))
        backup = MockLLM(is_chat_model=True)
        fallback = FallbackLLM(llms=[primary, backup])
        response = fallback.chat(
            [ChatMessage(role=MessageRole.USER, content="hello")]
        )
        assert isinstance(response, ChatResponse)

    def test_no_fallback_on_permanent_error(self) -> None:
        primary = _make_failing_mock(["complete"], Exception("invalid api key"))
        backup = MockLLM()
        # The fallback should never be called for permanent errors
        fallback = FallbackLLM(llms=[primary, backup])
        with pytest.raises(Exception, match="invalid api key"):
            fallback.complete("hello")

    def test_retries_before_fallback(self) -> None:
        """Provider is retried *retry_attempts* times before falling back."""
        primary = _make_failing_mock(["complete"], TimeoutError("timed out"))
        backup = MockLLM()
        fallback = FallbackLLM(llms=[primary, backup], retry_attempts=3)
        response = fallback.complete("hello")
        assert isinstance(response, CompletionResponse)
        assert primary.complete.call_count == 3  # type: ignore[attr-defined]

    def test_all_providers_exhausted(self) -> None:
        llm1 = _make_failing_mock(["complete"], TimeoutError("timeout 1"))
        llm2 = _make_failing_mock(["complete"], TimeoutError("timeout 2"))
        fallback = FallbackLLM(llms=[llm1, llm2])
        with pytest.raises(RuntimeError, match="all 2 provider"):
            fallback.complete("hello")

    def test_stream_fallback_on_transient(self) -> None:
        primary = _make_failing_mock(
            ["stream_complete"], TimeoutError("timed out")
        )
        backup = MockLLM()
        fallback = FallbackLLM(llms=[primary, backup])
        gen = fallback.stream_complete("hello")
        tokens = list(gen)
        assert isinstance(tokens, list)


# ---------------------------------------------------------------------------
# FallbackLLM -- failover (async)
# ---------------------------------------------------------------------------


class TestFallbackLLMFailoverAsync:
    @pytest.mark.asyncio()
    async def test_falls_back_on_transient_acomplete(self) -> None:
        primary = _make_failing_mock(["acomplete"], TimeoutError("timed out"))
        backup = MockLLM()
        fallback = FallbackLLM(llms=[primary, backup])
        response = await fallback.acomplete("hello")
        assert isinstance(response, CompletionResponse)

    @pytest.mark.asyncio()
    async def test_falls_back_on_transient_achat(self) -> None:
        primary = _make_failing_mock(["achat"], ConnectionError("connection refused"))
        backup = MockLLM(is_chat_model=True)
        fallback = FallbackLLM(llms=[primary, backup])
        response = await fallback.achat(
            [ChatMessage(role=MessageRole.USER, content="hello")]
        )
        assert isinstance(response, ChatResponse)

    @pytest.mark.asyncio()
    async def test_no_fallback_on_permanent_error_async(self) -> None:
        primary = _make_failing_mock(["acomplete"], Exception("invalid api key"))
        backup = MockLLM()
        fallback = FallbackLLM(llms=[primary, backup])
        with pytest.raises(Exception, match="invalid api key"):
            await fallback.acomplete("hello")

    @pytest.mark.asyncio()
    async def test_astream_fallback_on_transient(self) -> None:
        primary = _make_failing_mock(
            ["astream_complete"], TimeoutError("timed out")
        )
        backup = MockLLM()
        fallback = FallbackLLM(llms=[primary, backup])
        agen = await fallback.astream_complete("hello")
        tokens = [t async for t in agen]
        assert isinstance(tokens, list)
