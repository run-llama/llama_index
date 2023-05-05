"""
Github readers utils.

This module contains utility functions for the Github readers.
"""
import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

from llama_index.readers.github_readers.github_api_client import (
    GitBlobResponseModel,
    GithubClient,
    GitTreeResponseModel,
)


def print_if_verbose(verbose: bool, message: str) -> None:
    """Log message if verbose is True."""
    if verbose:
        print(message)


def get_file_extension(filename: str) -> str:
    """Get file extension."""
    return f".{os.path.splitext(filename)[1][1:].lower()}"


class BufferedAsyncIterator(ABC):
    """
    Base class for buffered async iterators.

    This class is to be used as a base class for async iterators
    that need to buffer the results of an async operation.
    The async operation is defined in the _fill_buffer method.
    The _fill_buffer method is called when the buffer is empty.
    """

    def __init__(self, buffer_size: int):
        """
        Initialize params.

        Args:
            - `buffer_size (int)`: Size of the buffer.
                It is also the number of items that will
                be retrieved from the async operation at once.
                see _fill_buffer. Defaults to 2. Setting it to 1
                will result in the same behavior as a synchronous iterator.
        """
        self._buffer_size = buffer_size
        self._buffer: List[Tuple[GitBlobResponseModel, str]] = []
        self._index = 0

    @abstractmethod
    async def _fill_buffer(self) -> None:
        raise NotImplementedError

    def __aiter__(self) -> "BufferedAsyncIterator":
        """Return the iterator object."""
        return self

    async def __anext__(self) -> Tuple[GitBlobResponseModel, str]:
        """
        Get next item.

        Returns:
            - `item (Tuple[GitBlobResponseModel, str])`: Next item.

        Raises:
            - `StopAsyncIteration`: If there are no more items.
        """
        if not self._buffer:
            await self._fill_buffer()

        if not self._buffer:
            raise StopAsyncIteration

        item = self._buffer.pop(0)
        self._index += 1
        return item


class BufferedGitBlobDataIterator(BufferedAsyncIterator):
    """
    Buffered async iterator for Git blobs.

    This class is an async iterator that buffers the results of the get_blob operation.
    It is used to retrieve the contents of the files in a Github repository.
    getBlob endpoint supports up to 100 megabytes of content for blobs.
    This concrete implementation of BufferedAsyncIterator allows you to lazily retrieve
    the contents of the files in a Github repository.
    Otherwise you would have to retrieve all the contents of
    the files in the repository at once, which would
    be problematic if the repository is large.
    """

    def __init__(
        self,
        blobs_and_paths: List[Tuple[GitTreeResponseModel.GitTreeObject, str]],
        github_client: GithubClient,
        owner: str,
        repo: str,
        loop: asyncio.AbstractEventLoop,
        buffer_size: int,
        verbose: bool = False,
    ):
        """
        Initialize params.

        Args:
            - blobs_and_paths (List[Tuple[GitTreeResponseModel.GitTreeObject, str]]):
                List of tuples containing the blob and the path of the file.
            - github_client (GithubClient): Github client.
            - owner (str): Owner of the repository.
            - repo (str): Name of the repository.
            - loop (asyncio.AbstractEventLoop): Event loop.
            - buffer_size (int): Size of the buffer.
        """
        super().__init__(buffer_size)
        self._blobs_and_paths = blobs_and_paths
        self._github_client = github_client
        self._owner = owner
        self._repo = repo
        self._verbose = verbose
        if loop is None:
            loop = asyncio.get_event_loop()
            if loop is None:
                raise ValueError("No event loop found")

    async def _fill_buffer(self) -> None:
        """
        Fill the buffer with the results of the get_blob operation.

        The get_blob operation is called for each blob in the blobs_and_paths list.
        The blobs are retrieved in batches of size buffer_size.
        """
        del self._buffer[:]
        self._buffer = []
        start = self._index
        end = min(start + self._buffer_size, len(self._blobs_and_paths))

        if start >= end:
            return

        if self._verbose:
            start_t = time.time()
        results: List[GitBlobResponseModel] = await asyncio.gather(
            *[
                self._github_client.get_blob(self._owner, self._repo, blob.sha)
                for blob, _ in self._blobs_and_paths[
                    start:end
                ]  # TODO: use batch_size instead of buffer_size for concurrent requests
            ]
        )
        if self._verbose:
            end_t = time.time()
            blob_names_and_sizes = [
                (blob.path, blob.size) for blob, _ in self._blobs_and_paths[start:end]
            ]
            print(
                "Time to get blobs ("
                + f"{blob_names_and_sizes}"
                + f"): {end_t - start_t:.2f} seconds"
            )

        self._buffer = [
            (result, path)
            for result, (_, path) in zip(results, self._blobs_and_paths[start:end])
        ]
