"""
Tests for issue #19367: GitHub readers should work in async contexts.

Previously, GitHubRepositoryIssuesReader, GitHubRepositoryCollaboratorsReader,
and GithubRepositoryReader used `self._loop.run_until_complete()` which raises
`RuntimeError: This event loop is already running` when called from within an
async context (e.g. FastAPI, MCP servers, Jupyter notebooks).

The fix replaces all `run_until_complete` calls with `asyncio_run` from
`llama_index.core.async_utils`, which handles running loops by dispatching
to a separate thread.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from llama_index.readers.github import GithubRepositoryReader
from llama_index.readers.github.repository.github_client import (
    GitBlobResponseModel,
    GitBranchResponseModel,
    GitTreeResponseModel,
    GithubClient,
)
from llama_index.readers.github.issues.github_client import GitHubIssuesClient
from llama_index.readers.github.collaborators.github_client import (
    GitHubCollaboratorsClient,
)
from llama_index.readers.github import (
    GitHubRepositoryIssuesReader,
    GitHubRepositoryCollaboratorsReader,
)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "fake-token-for-tests")

BRANCH_JSON = '{"name":"main","commit":{"sha":"abc123","node_id":"C_test","commit":{"author":{"name":"Test","email":"test@test.com","date":"2024-01-01T00:00:00Z"},"committer":{"name":"Test","email":"test@test.com","date":"2024-01-01T00:00:00Z"},"message":"test commit","tree":{"sha":"tree123","url":"https://api.github.com/test/trees/tree123"},"url":"https://api.github.com/test/commits/abc123","comment_count":0,"verification":{"verified":false,"reason":"unsigned","signature":null,"payload":null}},"url":"https://api.github.com/test/commits/abc123","html_url":"https://github.com/test/test/commit/abc123","comments_url":"https://api.github.com/test/commits/abc123/comments","author":null,"committer":null,"parents":[]},"_links":{"self":"https://api.github.com/test/branches/main","html":"https://github.com/test/test/tree/main"},"protected":false}'

TREE_JSON = '{"sha":"tree123","url":"https://api.github.com/test/trees/tree123","tree":[{"path":"README.md","mode":"100644","type":"blob","sha":"blob123","size":10,"url":"https://api.github.com/test/blobs/blob123"}],"truncated":false}'

BLOB_JSON = '{"sha":"blob123","node_id":"B_test","size":10,"url":"https://api.github.com/test/blobs/blob123","content":"SGVsbG8gV29ybGQ=","encoding":"base64"}'


@pytest.fixture()
def mock_github_client(monkeypatch):
    """Mock GithubClient API methods to avoid real network calls."""

    async def mock_get_branch(self, *args, **kwargs):
        return GitBranchResponseModel.from_json(BRANCH_JSON)

    async def mock_get_tree(self, *args, **kwargs):
        return GitTreeResponseModel.from_json(TREE_JSON)

    async def mock_get_blob(self, *args, **kwargs):
        return GitBlobResponseModel.from_json(BLOB_JSON)

    monkeypatch.setattr(GithubClient, "get_branch", mock_get_branch)
    monkeypatch.setattr(GithubClient, "get_tree", mock_get_tree)
    monkeypatch.setattr(GithubClient, "get_blob", mock_get_blob)


def test_github_repository_reader_no_loop_attribute():
    """GithubRepositoryReader.__init__ must not create self._loop."""
    client = GithubClient(GITHUB_TOKEN)
    reader = GithubRepositoryReader(client, "owner", "repo")
    assert not hasattr(reader, "_loop"), (
        "GithubRepositoryReader should not have _loop attribute after fix"
    )


def test_github_issues_reader_no_loop_attribute():
    """GitHubRepositoryIssuesReader.__init__ must not create self._loop."""
    client = GitHubIssuesClient(github_token=GITHUB_TOKEN)
    reader = GitHubRepositoryIssuesReader(
        github_client=client, owner="owner", repo="repo"
    )
    assert not hasattr(reader, "_loop"), (
        "GitHubRepositoryIssuesReader should not have _loop attribute after fix"
    )


def test_github_collaborators_reader_no_loop_attribute():
    """GitHubRepositoryCollaboratorsReader.__init__ must not create self._loop."""
    client = GitHubCollaboratorsClient(github_token=GITHUB_TOKEN)
    reader = GitHubRepositoryCollaboratorsReader(
        github_client=client, owner="owner", repo="repo"
    )
    assert not hasattr(reader, "_loop"), (
        "GitHubRepositoryCollaboratorsReader should not have _loop attribute after fix"
    )


def test_repository_reader_load_data_within_running_loop(mock_github_client):
    """
    GithubRepositoryReader.load_data() must succeed when called from within
    a running event loop (simulating FastAPI / MCP server / Jupyter context).

    Before the fix this raised:
        RuntimeError: This event loop is already running
    """

    async def run_in_async_context():
        client = GithubClient(GITHUB_TOKEN)
        reader = GithubRepositoryReader(client, "owner", "repo")
        # load_data is synchronous but internally calls async code;
        # with asyncio_run it must handle the already-running loop.
        return reader.load_data(branch="main")

    documents = asyncio.run(run_in_async_context())
    assert isinstance(documents, list)


def test_issues_reader_load_data_within_running_loop():
    """
    GitHubRepositoryIssuesReader.load_data() must succeed in async context.
    """
    empty_page: list = []

    async def mock_get_issues(self, *args, **kwargs):
        return empty_page

    async def run_in_async_context():
        with patch.object(GitHubIssuesClient, "get_issues", mock_get_issues):
            client = GitHubIssuesClient(github_token=GITHUB_TOKEN)
            reader = GitHubRepositoryIssuesReader(
                github_client=client, owner="owner", repo="repo"
            )
            return reader.load_data()

    documents = asyncio.run(run_in_async_context())
    assert documents == []


def test_collaborators_reader_load_data_within_running_loop():
    """
    GitHubRepositoryCollaboratorsReader.load_data() must succeed in async context.
    """
    empty_page: list = []

    async def mock_get_collaborators(self, *args, **kwargs):
        return empty_page

    async def run_in_async_context():
        with patch.object(
            GitHubCollaboratorsClient, "get_collaborators", mock_get_collaborators
        ):
            client = GitHubCollaboratorsClient(github_token=GITHUB_TOKEN)
            reader = GitHubRepositoryCollaboratorsReader(
                github_client=client, owner="owner", repo="repo"
            )
            return reader.load_data()

    documents = asyncio.run(run_in_async_context())
    assert documents == []


def test_buffered_git_blob_iterator_loop_param_deprecation():
    """
    BufferedGitBlobDataIterator should emit DeprecationWarning when 'loop'
    is passed, for backwards compatibility.
    """
    import warnings
    from llama_index.readers.github.repository.utils import BufferedGitBlobDataIterator

    client = GithubClient(GITHUB_TOKEN)
    loop = asyncio.new_event_loop()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BufferedGitBlobDataIterator(
                blobs_and_paths=[],
                github_client=client,
                owner="owner",
                repo="repo",
                buffer_size=5,
                loop=loop,
            )
        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 1
        assert "loop" in str(deprecation_warnings[0].message).lower()
    finally:
        loop.close()


def test_buffered_git_blob_iterator_no_loop_param_no_warning():
    """
    BufferedGitBlobDataIterator should not emit DeprecationWarning when
    'loop' is not passed (the normal post-fix usage).
    """
    import warnings
    from llama_index.readers.github.repository.utils import BufferedGitBlobDataIterator

    client = GithubClient(GITHUB_TOKEN)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BufferedGitBlobDataIterator(
            blobs_and_paths=[],
            github_client=client,
            owner="owner",
            repo="repo",
            buffer_size=5,
        )
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 0
