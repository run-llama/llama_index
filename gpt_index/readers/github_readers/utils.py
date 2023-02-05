import asyncio
from abc import ABC, abstractmethod
from typing import List, Tuple

from gpt_index.readers.github_readers.github_api_client import (
    GitBlobResponseModel,
    GithubClient,
    GitTreeResponseModel,
)


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


class BufferedGitBlobDataIterator(BufferedAsyncIterator):
    def __init__(
        self,
        blobs_and_paths: List[Tuple[GitTreeResponseModel.GitTreeObject, str]],
        github_client: GithubClient,
        owner: str,
        repo: str,
        loop: asyncio.AbstractEventLoop,
        buffer_size: int,
    ):
        super().__init__(buffer_size)
        self.blobs_and_paths = blobs_and_paths
        self.github_client = github_client
        self.owner = owner
        self.repo = repo
        if loop is None:
            loop = asyncio.get_event_loop()
            if loop is None:
                raise ValueError("No event loop found")

    async def _fill_buffer(self):
        del self.buffer[:]
        self.buffer = []
        start = self.index
        end = min(start + self.buffer_size, len(self.blobs_and_paths))

        if start >= end:
            return

        results: List[GitBlobResponseModel] = await asyncio.gather(
            *[
                self.github_client.get_blob(self.owner, self.repo, blob.sha)
                for blob, _ in self.blobs_and_paths[start:end]
            ]
        )

        self.buffer = [
            (result, path)
            for result, (_, path) in zip(results, self.blobs_and_paths[start:end])
        ]
