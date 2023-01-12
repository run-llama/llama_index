"""Test utils."""

from typing import Optional, Type

import pytest

from gpt_index.utils import globals_helper, retry_on_exceptions_with_backoff


def test_tokenizer() -> None:
    """Make sure tokenizer works.

    NOTE: we use a different tokenizer for python >= 3.9.

    """
    text = "hello world foo bar"
    tokenizer = globals_helper.tokenizer
    assert len(tokenizer(text)) == 4


call_count = 0


def fn_with_exception(exception_cls: Optional[Type[Exception]]) -> bool:
    """Return true unless exception if specified."""
    global call_count
    call_count += 1
    if exception_cls:
        raise exception_cls
    return True


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
            [ValueError],
            max_tries=3,
            min_backoff_secs=0.0,
        )
    assert call_count == 3

    # different exception will not get retried
    call_count = 0
    with pytest.raises(TypeError):
        retry_on_exceptions_with_backoff(
            lambda: fn_with_exception(TypeError),
            [ValueError],
            max_tries=3,
        )
    assert call_count == 1
