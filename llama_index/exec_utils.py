import copy

from typing import Any, Mapping
from types import CodeType
from _typeshed import ReadableBuffer


def safe_eval(
    __source: str | ReadableBuffer | CodeType,
    __globals: dict[str, Any] | None = None,
    __locals: Mapping[str, object] | None = None,
) -> Any:
    pass


def safe_exec(
    __source: str | ReadableBuffer | CodeType,
    __globals: dict[str, Any] | None = None,
    __locals: Mapping[str, object] | None = None,
) -> None:
    pass
