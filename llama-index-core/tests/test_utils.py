"""Test utils."""

from pathlib import Path
from typing import Optional, Type, Union
from unittest import mock

import pytest
from _pytest.capture import CaptureFixture
from llama_index.core.utils import (
    _ANSI_COLORS,
    _LLAMA_INDEX_COLORS,
    ErrorToRetry,
    _get_colored_text,
    get_cache_dir,
    get_color_mapping,
    get_retry_on_exceptions_with_backoff_decorator,
    get_tokenizer,
    iter_batch,
    print_text,
    retry_on_exceptions_with_backoff,
)


def test_tokenizer() -> None:
    """
    Make sure tokenizer works.

    NOTE: we use a different tokenizer for python >= 3.9.

    """
    text = "hello world foo bar"
    tokenizer = get_tokenizer()
    assert len(tokenizer(text)) == 4


call_count = 0


def fn_with_exception(
    exception_cls: Optional[Union[Type[Exception], Exception]],
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


@pytest.mark.asyncio
async def test_retry_on_exceptions_with_backoff_decorator() -> None:
    """Make sure retry decorator works for both sync and async functions."""
    global call_count
    call_count = 0

    retry_on_value_error = get_retry_on_exceptions_with_backoff_decorator(
        [ErrorToRetry(ValueError)],
        max_tries=3,
        min_backoff_secs=0.0,
    )

    SUCCESS_MESSAGE = "done"

    @retry_on_value_error
    def fn_with_exception(exception, n=2) -> None:
        global call_count
        call_count += 1
        if call_count >= n:
            return SUCCESS_MESSAGE
        raise exception

    @retry_on_value_error
    async def async_fn_with_exception(exception, n=2) -> None:
        global call_count
        call_count += 1
        if call_count >= n:
            return SUCCESS_MESSAGE
        raise exception

    # sync function
    # should retry 3 times
    call_count = 0
    with pytest.raises(ValueError):
        result = fn_with_exception(ValueError, 5)
    assert call_count == 3

    # should not raise exception
    call_count = 0
    result = fn_with_exception(ValueError, 2)
    assert result == SUCCESS_MESSAGE
    assert call_count == 2

    # different exception will not get retried
    call_count = 0
    with pytest.raises(TypeError):
        result = fn_with_exception(TypeError, 2)
    assert call_count == 1

    # Async function
    # should retry 3 times
    call_count = 0
    with pytest.raises(ValueError):
        result = await async_fn_with_exception(ValueError, 5)
    assert call_count == 3

    # should not raise exception
    call_count = 0
    result = await async_fn_with_exception(ValueError, 2)
    assert result == SUCCESS_MESSAGE
    assert call_count == 2

    # different exception will not get retried
    call_count = 0
    with pytest.raises(TypeError):
        result = await async_fn_with_exception(TypeError, 2)
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


def test_get_cache_dir_with_env_override(tmp_path, monkeypatch) -> None:
    custom_cache_dir = str(tmp_path / "custom_cache")

    # Test with environment variable set
    monkeypatch.setenv("LLAMA_INDEX_CACHE_DIR", custom_cache_dir)
    result = get_cache_dir()
    assert result == custom_cache_dir
    assert isinstance(result, str)
    assert Path(custom_cache_dir).exists()
    assert Path(custom_cache_dir).is_dir()


def test_get_cache_dir_default_behavior(monkeypatch) -> None:
    # Ensure environment variable is not set
    monkeypatch.delenv("LLAMA_INDEX_CACHE_DIR", raising=False)

    with mock.patch("platformdirs.user_cache_dir") as mock_user_cache_dir:
        mock_cache_path = "/mock/cache/llama_index"
        mock_user_cache_dir.return_value = mock_cache_path

        with mock.patch("pathlib.Path.mkdir") as mock_mkdir:
            result = get_cache_dir()
            mock_user_cache_dir.assert_called_once_with("llama_index")
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            assert result == mock_cache_path


def test_get_cache_dir_creates_directory(tmp_path, monkeypatch) -> None:
    cache_dir = str(tmp_path / "test_cache" / "nested" / "llama_index")

    # Ensure directory doesn't exist initially
    assert not Path(cache_dir).exists()
    monkeypatch.setenv("LLAMA_INDEX_CACHE_DIR", cache_dir)
    result = get_cache_dir()

    assert Path(cache_dir).exists()
    assert Path(cache_dir).is_dir()
    assert result == cache_dir


def test_get_cache_dir_no_toctou_issue(tmp_path, monkeypatch) -> None:
    cache_dir = str(tmp_path / "toctou_test")
    monkeypatch.setenv("LLAMA_INDEX_CACHE_DIR", cache_dir)
    with mock.patch("pathlib.Path.mkdir") as mock_mkdir:
        get_cache_dir()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_get_cache_dir_env_var_precedence(tmp_path, monkeypatch) -> None:
    env_cache_dir = str(tmp_path / "env_cache")
    monkeypatch.setenv("LLAMA_INDEX_CACHE_DIR", env_cache_dir)

    with mock.patch("platformdirs.user_cache_dir") as mock_user_cache_dir:
        mock_user_cache_dir.return_value = "/should/not/be/used"
        get_cache_dir()
        mock_user_cache_dir.assert_not_called()
