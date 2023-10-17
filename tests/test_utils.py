"""Test utils."""

from typing import Optional, Type, Union

import pytest
from _pytest.capture import CaptureFixture
from llama_index.utils import (
    _ANSI_COLORS,
    _LLAMA_INDEX_COLORS,
    ErrorToRetry,
    _get_colored_text,
    get_color_mapping,
    globals_helper,
    iter_batch,
    print_text,
    retry_on_exceptions_with_backoff,
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
    lst = list(range(6))
    assert list(iter_batch(lst, 3)) == [[0, 1, 2], [3, 4, 5]]

    gen = (i for i in range(5))
    assert list(iter_batch(gen, 3)) == [[0, 1, 2], [3, 4]]

    assert list(iter_batch([], 3)) == []


def test_get_color_mapping() -> None:
    """Test get_color_mapping function."""
    items = ["item1", "item2", "item3", "item4"]
    color_mapping = get_color_mapping(items)
    assert len(color_mapping) == len(items)
    assert set(color_mapping.keys()) == set(items)
    assert all(color in _LLAMA_INDEX_COLORS for color in color_mapping.values())

    color_mapping_ansi = get_color_mapping(items, use_llama_index_colors=False)
    assert len(color_mapping_ansi) == len(items)
    assert set(color_mapping_ansi.keys()) == set(items)
    assert all(color in _ANSI_COLORS for color in color_mapping_ansi.values())


def test_get_colored_text() -> None:
    """Test _get_colored_text function."""
    text = "Hello, world!"
    for color in _LLAMA_INDEX_COLORS:
        colored_text = _get_colored_text(text, color)
        assert colored_text.startswith("\033[1;3;")
        assert colored_text.endswith("m" + text + "\033[0m")

    for color in _ANSI_COLORS:
        colored_text = _get_colored_text(text, color)
        assert colored_text.startswith("\033[1;3;")
        assert colored_text.endswith("m" + text + "\033[0m")

    # Test with an unsupported color
    colored_text = _get_colored_text(text, "unsupported_color")
    assert colored_text == f"\033[1;3m{text}\033[0m"  # just bolded and italicized


def test_print_text(capsys: CaptureFixture) -> None:
    """Test print_text function."""
    text = "Hello, world!"
    for color in _LLAMA_INDEX_COLORS:
        print_text(text, color)
        captured = capsys.readouterr()
        assert captured.out == f"\033[1;3;{_LLAMA_INDEX_COLORS[color]}m{text}\033[0m"

    for color in _ANSI_COLORS:
        print_text(text, color)
        captured = capsys.readouterr()
        assert captured.out == f"\033[1;3;{_ANSI_COLORS[color]}m{text}\033[0m"

    # Test with an unsupported color
    print_text(text, "unsupported_color")
    captured = capsys.readouterr()
    assert captured.out == f"\033[1;3m{text}\033[0m"

    # Test without color
    print_text(text)
    captured = capsys.readouterr()
    assert captured.out == f"{text}"

    # Test with end
    print_text(text, end=" ")
    captured = capsys.readouterr()
    assert captured.out == f"{text} "
