from __future__ import annotations

import json
import pytest

from llama_index.readers.github import GithubRepositoryReader
from llama_index.readers.github.repository.github_client import (
    GithubClient,
    GitContentResponseModel,
    GitCommitResponseModel,
    GitTreeResponseModel,
)

# Mock Data
CONTENT_JSON = json.dumps(
    {
        "type": "file",
        "encoding": "base64",
        "size": 12,
        "name": "test_file.txt",
        "path": "test_file.txt",
        "content": "VGVzdCBjb250ZW50",  # "Test content"
        "sha": "test_sha",
        "url": "https://api.github.com/repos/run-llama/llama_index/contents/test_file.txt",
        "git_url": "https://api.github.com/repos/run-llama/llama_index/git/blobs/test_sha",
        "html_url": "https://github.com/run-llama/llama_index/blob/main/test_file.txt",
        "download_url": "https://raw.githubusercontent.com/run-llama/llama_index/main/test_file.txt",
        "_links": {"self": "...", "git": "...", "html": "..."},
    }
)

COMMIT_JSON = (
    '{"sha":"test_commit_sha","url":"...","commit":{"tree":{"sha":"test_tree_sha"}}}'
)
TREE_JSON = json.dumps(
    {
        "sha": "test_tree_sha",
        "url": "...",
        "tree": [
            {
                "path": "test_file.txt",
                "mode": "100644",
                "type": "blob",
                "sha": "test_sha",
                "size": 12,
                "url": "...",
            }
        ],
        "truncated": False,
    }
)
BLOB_JSON = json.dumps(
    {
        "sha": "test_sha",
        "size": 12,
        "url": "...",
        "content": "VGVzdCBjb250ZW50",
        "encoding": "base64",
        "node_id": "test_node_id",
    }
)


@pytest.fixture
def mock_client(monkeypatch):
    async def mock_get_content(self, owner, repo, path, ref=None, **kwargs):
        # Return mock content for any path
        # In a real test, we might check path and ref
        return GitContentResponseModel.from_json(CONTENT_JSON)

    async def mock_get_commit(self, *args, **kwargs):
        return GitCommitResponseModel.from_json(COMMIT_JSON)

    async def mock_get_tree(self, *args, **kwargs):
        return GitTreeResponseModel.from_json(TREE_JSON)

    async def mock_get_blob(self, *args, **kwargs):
        from llama_index.readers.github.repository.github_client import (
            GitBlobResponseModel,
        )

        return GitBlobResponseModel.from_json(BLOB_JSON)

    monkeypatch.setattr(GithubClient, "get_content", mock_get_content)
    monkeypatch.setattr(GithubClient, "get_commit", mock_get_commit)
    monkeypatch.setattr(GithubClient, "get_tree", mock_get_tree)
    monkeypatch.setattr(GithubClient, "get_blob", mock_get_blob)

    return GithubClient("fake_token")


def test_load_specific_file_paths(mock_client):
    reader = GithubRepositoryReader(mock_client, "run-llama", "llama_index")

    # Test loading specific files via branch
    documents = reader.load_data(branch="main", file_paths=["test_file.txt"])

    assert len(documents) == 1
    assert documents[0].metadata["file_path"] == "test_file.txt"
    assert documents[0].text == "Test content"
    assert documents[0].id_ == "test_sha"


def test_load_specific_file_paths_deduplication(mock_client):
    reader = GithubRepositoryReader(mock_client, "run-llama", "llama_index")

    # Define check function that says "test_sha" exists
    def check_exists(sha):
        return sha == "test_sha"

    # expect empty list because check_exists returns True for the file's SHA
    documents = reader.load_data(
        branch="main", file_paths=["test_file.txt"], file_exists_callback=check_exists
    )

    assert len(documents) == 0


def test_scan_deduplication(mock_client):
    reader = GithubRepositoryReader(mock_client, "run-llama", "llama_index")

    # Normal scan (via commit or branch without file_paths)
    # Mock tree has 1 file with SHA "test_sha"

    # Case 1: No deduplication -> should get 1 doc
    documents = reader.load_data(commit_sha="test_commit_sha")
    assert len(documents) == 1

    # Case 2: Deduplication -> should get 0 docs
    def check_exists(sha):
        return sha == "test_sha"

    documents = reader.load_data(
        commit_sha="test_commit_sha", file_exists_callback=check_exists
    )
    assert len(documents) == 0
