from __future__ import annotations

from llama_index.readers.github import GithubRepositoryReader
from llama_index.readers.github.repository.github_client import GithubClient
from llama_index.core.schema import Document
import os
import pytest
from httpx import HTTPError


@pytest.fixture()
def mock_error(monkeypatch):
    async def mock_get_blob(self, *args, **kwargs):
        if self._fail_on_http_error:
            raise HTTPError("Woops")
        else:
            return

    monkeypatch.setattr(GithubClient, "get_blob", mock_get_blob)


def test_fail_on_http_error_true(mock_error):
    token = os.getenv("GITHUB_TOKEN")
    gh_client = GithubClient(token, fail_on_http_error=True)
    reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")
    with pytest.raises(HTTPError):
        reader.load_data(branch="main")


def test_fail_on_http_error_false(mock_error):
    token = os.getenv("GITHUB_TOKEN")
    gh_client = GithubClient(token, fail_on_http_error=False)
    reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")
    documents = reader.load_data(branch="main")
    assert isinstance(documents, list)
    assert all(isinstance(doc, Document) for doc in documents)
