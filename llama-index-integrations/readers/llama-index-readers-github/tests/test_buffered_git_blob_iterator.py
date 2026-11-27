"""
Tests for BufferedGitBlobDataIterator's blob/path pairing.

``GithubClient.get_blob`` returns ``None`` when a blob cannot be fetched
(HTTP error with ``fail_on_http_error=False``), so the buffered iterator has to
keep every fetched blob attached to its own path and skip the failed ones.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from llama_index.readers.github.repository.utils import BufferedGitBlobDataIterator


@dataclass
class _FakeBlob:
    sha: str
    content: str = ""
    path: str = ""
    size: int = 1


class _FakeGithubClient:
    """Mirrors ``GithubClient.get_blob``: returns ``None`` for a failed fetch."""

    def __init__(self, contents: Dict[str, str], failing_shas: Sequence[str] = ()):
        self._contents = contents
        self._failing_shas = set(failing_shas)
        self.requested: List[str] = []

    async def get_blob(
        self, owner: str, repo: str, file_sha: str, timeout: int = 5, retries: int = 0
    ) -> Optional[_FakeBlob]:
        self.requested.append(file_sha)
        if file_sha in self._failing_shas:
            return None
        return _FakeBlob(sha=file_sha, content=self._contents[file_sha])


def _collect(
    contents: Dict[str, str],
    failing_shas: Sequence[str] = (),
    buffer_size: int = 5,
) -> Tuple[List[Tuple[str, str]], _FakeGithubClient]:
    tree = [(_FakeBlob(sha=sha), f"{sha}.txt") for sha in contents]
    client = _FakeGithubClient(contents, failing_shas)

    async def run() -> List[Tuple[str, str]]:
        iterator = BufferedGitBlobDataIterator(
            blobs_and_paths=tree,
            github_client=client,
            owner="owner",
            repo="repo",
            buffer_size=buffer_size,
        )
        return [(blob.content, path) async for blob, path in iterator]

    return asyncio.run(run()), client


def test_failed_blob_does_not_shift_paths() -> None:
    contents = {"sha_a": "a", "sha_b": "b", "sha_c": "c"}

    emitted, _ = _collect(contents, failing_shas=["sha_b"])

    assert emitted == [("a", "sha_a.txt"), ("c", "sha_c.txt")]


def test_failed_blob_batches_are_not_fetched_twice() -> None:
    contents = {"sha_a": "a", "sha_b": "b", "sha_c": "c", "sha_d": "d"}

    emitted, client = _collect(contents, failing_shas=["sha_b"], buffer_size=2)

    assert [path for _, path in emitted] == ["sha_a.txt", "sha_c.txt", "sha_d.txt"]
    assert sorted(client.requested) == sorted(contents)


def test_all_blobs_are_emitted_when_every_fetch_succeeds() -> None:
    contents = {"sha_a": "a", "sha_b": "b", "sha_c": "c"}

    emitted, _ = _collect(contents, buffer_size=2)

    assert emitted == [("a", "sha_a.txt"), ("b", "sha_b.txt"), ("c", "sha_c.txt")]
