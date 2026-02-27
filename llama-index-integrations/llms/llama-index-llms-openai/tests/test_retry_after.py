from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from tenacity import Future, RetryCallState, wait_exponential

from llama_index.llms.openai.utils import (
    _MAX_RETRY_AFTER_SECONDS,
    _WaitRetryAfter,
    _parse_retry_after,
    create_retry_decorator,
)


def _make_rate_limit_error(headers=None):
    """Build an openai.RateLimitError with the given response headers."""
    response = httpx.Response(
        status_code=429,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return openai.RateLimitError(
        message="Rate limit exceeded",
        response=response,
        body=None,
    )


def _make_retry_state(exc):
    """Build a RetryCallState whose outcome holds the given exception."""
    rs = RetryCallState(
        retry_object=MagicMock(),
        fn=MagicMock(),
        args=(),
        kwargs={},
    )
    fut = Future(attempt_number=1)
    fut.set_exception(exc)
    rs.outcome = fut
    rs.attempt_number = 1
    return rs


# -- _parse_retry_after unit tests --


def test_parse_retry_after_integer():
    exc = _make_rate_limit_error(headers={"Retry-After": "30"})
    assert _parse_retry_after(exc) == 30.0


def test_parse_retry_after_float():
    exc = _make_rate_limit_error(headers={"Retry-After": "1.5"})
    assert _parse_retry_after(exc) == 1.5


def test_parse_retry_after_zero():
    exc = _make_rate_limit_error(headers={"Retry-After": "0"})
    assert _parse_retry_after(exc) == 0.0


def test_parse_retry_after_missing_header():
    exc = _make_rate_limit_error(headers={})
    assert _parse_retry_after(exc) is None


def test_parse_retry_after_non_numeric():
    exc = _make_rate_limit_error(headers={"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"})
    assert _parse_retry_after(exc) is None


def test_parse_retry_after_negative():
    exc = _make_rate_limit_error(headers={"Retry-After": "-5"})
    assert _parse_retry_after(exc) is None


def test_parse_retry_after_empty_string():
    exc = _make_rate_limit_error(headers={"Retry-After": ""})
    assert _parse_retry_after(exc) is None


def test_parse_retry_after_no_response():
    exc = openai.RateLimitError.__new__(openai.RateLimitError)
    assert _parse_retry_after(exc) is None


def test_parse_retry_after_case_insensitive():
    """httpx.Headers is case-insensitive, so 'RETRY-AFTER' should work."""
    exc = _make_rate_limit_error(headers={"RETRY-AFTER": "42"})
    assert _parse_retry_after(exc) == 42.0


# -- _WaitRetryAfter unit tests --


def test_wait_retry_after_uses_header():
    fallback = wait_exponential(multiplier=1, min=4, max=60)
    strategy = _WaitRetryAfter(fallback)

    exc = _make_rate_limit_error(headers={"Retry-After": "15"})
    rs = _make_retry_state(exc)
    assert strategy(rs) == 15.0


def test_wait_retry_after_caps_at_maximum():
    fallback = wait_exponential(multiplier=1, min=4, max=60)
    strategy = _WaitRetryAfter(fallback)

    exc = _make_rate_limit_error(headers={"Retry-After": "9999"})
    rs = _make_retry_state(exc)
    assert strategy(rs) == _MAX_RETRY_AFTER_SECONDS


def test_wait_retry_after_falls_back_when_no_header():
    fallback = MagicMock(return_value=5.0)
    strategy = _WaitRetryAfter(fallback)

    exc = _make_rate_limit_error(headers={})
    rs = _make_retry_state(exc)
    assert strategy(rs) == 5.0
    fallback.assert_called_once_with(rs)


def test_wait_retry_after_falls_back_for_non_rate_limit_error():
    fallback = MagicMock(return_value=7.0)
    strategy = _WaitRetryAfter(fallback)

    exc = openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))
    rs = _make_retry_state(exc)
    assert strategy(rs) == 7.0
    fallback.assert_called_once_with(rs)


def test_wait_retry_after_falls_back_when_header_unparseable():
    fallback = MagicMock(return_value=6.0)
    strategy = _WaitRetryAfter(fallback)

    exc = _make_rate_limit_error(headers={"Retry-After": "not-a-number"})
    rs = _make_retry_state(exc)
    assert strategy(rs) == 6.0
    fallback.assert_called_once_with(rs)


def test_wait_retry_after_falls_back_when_outcome_is_none():
    fallback = MagicMock(return_value=4.0)
    strategy = _WaitRetryAfter(fallback)

    rs = RetryCallState(
        retry_object=MagicMock(),
        fn=MagicMock(),
        args=(),
        kwargs={},
    )
    rs.outcome = None
    assert strategy(rs) == 4.0
    fallback.assert_called_once_with(rs)


# -- create_retry_decorator integration tests --


def test_create_retry_decorator_respects_retry_after():
    """Verify the full decorator stack uses Retry-After when available."""
    call_count = 0

    @create_retry_decorator(max_retries=3)
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _make_rate_limit_error(headers={"Retry-After": "0"})
        return "ok"

    with patch("llama_index.llms.openai.utils.logger"):
        result = flaky_function()

    assert result == "ok"
    assert call_count == 3


def test_create_retry_decorator_exhausts_retries():
    """Verify retries stop at max_retries even with Retry-After."""

    @create_retry_decorator(max_retries=2)
    def always_fails():
        raise _make_rate_limit_error(headers={"Retry-After": "0"})

    with (
        patch("llama_index.llms.openai.utils.logger"),
        pytest.raises(openai.RateLimitError),
    ):
        always_fails()


def test_create_retry_decorator_non_rate_limit_still_retries():
    """Non-RateLimitError exceptions still retry with exponential backoff."""
    call_count = 0

    @create_retry_decorator(max_retries=3)
    def timeout_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise openai.APITimeoutError(
                request=httpx.Request("POST", "https://api.openai.com")
            )
        return "ok"

    with patch("llama_index.llms.openai.utils.logger"):
        result = timeout_then_succeed()

    assert result == "ok"
    assert call_count == 2
