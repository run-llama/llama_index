"""Test utils."""

from typing import Optional, Type, Union

import pytest

from llama_index.utils import (
    ErrorToRetry,
    globals_helper,
    retry_on_exceptions_with_backoff,
    iter_batch,
)


def test_tokenizer() -> None:
    """Make sure tokenizer works.

    NOTE: we use a different tokenizer for python >= 3.9.

    """
    text = "hello world foo bar"
    tokenizer = globals_helper.tokenizer
    assert len(tokenizer(text)) == 4


call_count = 0


def fn_with_exception(
    exception_cls: Optional[Union[Type[Exception], Exception]]
) -> bool:
    """Return true unless exception is specified."""
    global call_count
    call_count += 1
    if exception_cls:
        raise exception_cls
    return True


class ConditionalException(Exception):
    """Exception that contains retry attribute."""

    def __init__(self, should_retry: bool) -> None:
        """Initialize with parameters."""
        self.should_retry = should_retry


def test_retry_on_exceptions_with_backoff() -> None:
    """Make sure retry function has accurate number of attempts."""
    global call_count
    assert fn_with_exception(None)

    call_count = 0
    with pytest.raises(ValueError):
        fn_with_exception(ValueError)
    assert call_count == 1

    call_count = 0
    with pytest.raises(ValueError):
        retry_on_exceptions_with_backoff(
            lambda: fn_with_exception(ValueError),
            [ErrorToRetry(ValueError)],
            max_tries=3,
            min_backoff_secs=0.0,
        )
    assert call_count == 3

    # different exception will not get retried
    call_count = 0
    with pytest.raises(TypeError):
        retry_on_exceptions_with_backoff(
            lambda: fn_with_exception(TypeError),
            [ErrorToRetry(ValueError)],
            max_tries=3,
        )
    assert call_count == 1


def test_retry_on_conditional_exceptions() -> None:
    """Make sure retry function works on conditional exceptions."""
    global call_count
    call_count = 0
    with pytest.raises(ConditionalException):
        retry_on_exceptions_with_backoff(
            lambda: fn_with_exception(ConditionalException(True)),
            [ErrorToRetry(ConditionalException, lambda e: e.should_retry)],
            max_tries=3,
            min_backoff_secs=0.0,
        )
    assert call_count == 3

    call_count = 0
    with pytest.raises(ConditionalException):
        retry_on_exceptions_with_backoff(
            lambda: fn_with_exception(ConditionalException(False)),
            [ErrorToRetry(ConditionalException, lambda e: e.should_retry)],
            max_tries=3,
            min_backoff_secs=0.0,
        )
    assert call_count == 1


def test_iter_batch() -> None:
    """Check iter_batch works as expected on regular, lazy and empty sequences."""
    lst = [i for i in range(6)]
    assert list(iter_batch(lst, 3)) == [[0, 1, 2], [3, 4, 5]]

    gen = (i for i in range(5))
    assert list(iter_batch(gen, 3)) == [[0, 1, 2], [3, 4]]

    assert list(iter_batch([], 3)) == []
