from __future__ import annotations

from typing import Any, Iterable


class AsyncIterator:
    def __init__(self, iterable: Iterable) -> None:
        self._iterable = iter(iterable)

    def __aiter__(self) -> AsyncIterator:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iterable)
        except StopIteration:
            raise StopAsyncIteration
