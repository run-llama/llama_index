from abc import ABC, abstractmethod
from typing import List, Tuple

from gpt_index.readers.github_readers.github_api_client import GitBlobResponseModel


def print_if_verbose(verbose: bool, message: str):
    """Log message if verbose is True."""
    if verbose:
        print(message)


class BufferedAsyncIterator(ABC):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer: List[Tuple[GitBlobResponseModel, str]] = []
        self.index = 0

    @abstractmethod
    async def _fill_buffer(self):
        raise NotImplementedError

    def __aiter__(self):
        return self

    async def __anext__(self) -> Tuple[GitBlobResponseModel, str]:
        if not self.buffer:
            await self._fill_buffer()

        if not self.buffer:
            raise StopAsyncIteration

        item = self.buffer.pop(0)
        self.index += 1
        return item
