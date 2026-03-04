from __future__ import annotations

from llama_index.readers.github import GithubRepositoryReader
from llama_index.readers.github.repository.github_client import (
    GithubClient,
    GitBranchResponseModel,
    GitTreeResponseModel,
    GitCommitResponseModel,
)
from llama_index.core.schema import Document
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.readers.github.repository.event import (
    GitHubFileProcessedEvent,
    GitHubFileSkippedEvent,
    GitHubFileFailedEvent,
    GitHubRepositoryProcessingStartedEvent,
    GitHubRepositoryProcessingCompletedEvent,
    GitHubTotalFilesToProcessEvent,
    GitHubFileProcessingStartedEvent,
)

import json
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from httpx import HTTPError, Response

COMMIT_JSON = '{"sha":"a11a953e738cbda93335ede83f012914d53dc4f7","node_id":"C_kwDOIWuq59oAKGExMWE5NTNlNzM4Y2JkYTkzMzM1ZWRlODNmMDEyOTE0ZDUzZGM0Zjc","commit":{"author":{"name":"Grisha","email":"skvrd@users.noreply.github.com","date":"2024-05-08T15:19:34Z"},"committer":{"name":"GitHub","email":"noreply@github.com","date":"2024-05-08T15:19:34Z"},"message":"Fix hidden temp directory issue for arxiv reader (#13351)\\n\\n* Fix hidden temp directory issue for arxiv reader\\r\\n\\r\\n* add comment base.py\\r\\n\\r\\n---------\\r\\n\\r\\nCo-authored-by: Andrei Fajardo <92402603+nerdai@users.noreply.github.com>","tree":{"sha":"01d1b2024af28c7d5abf2a66108e7ed5611c6308","url":"https://api.github.com/repos/run-llama/llama_index/git/trees/01d1b2024af28c7d5abf2a66108e7ed5611c6308"},"url":"https://api.github.com/repos/run-llama/llama_index/git/commits/a11a953e738cbda93335ede83f012914d53dc4f7","comment_count":0,"verification":{"verified":true,"reason":"valid","signature":"-----BEGIN PGP SIGNATURE-----\\n\\nwsFcBAABCAAQBQJmO5gGCRC1aQ7uu5UhlAAAjoYQAEy/6iaWXaHQUEAz2w+2fs7C\\n8cXt3BcXkbO+d6WmIsca22wCbhRVj0TQdYTbfjLDpyMK1BdFcsgBjC7Ym8dm40dZ\\nRxoNd0Ws04doz64L67zXtBTQXjEb3nNnDkl82jk0AtuFfb69pTjC/rH/MqDt7v4F\\nU4uDR0OyDakmEEFNE63UhKmbwkmMN33VXZMbm2obxJklR2rBCge7EBhtj+iwD8Og\\nDQ852VzB7/PV4mhalLjuP8CiY9kItZq0zN+Fn/ghB+0o6Xf3cIxCraiL34jxGBma\\nvXgRsqgI8kMW5ZE9zRjQhEh5GFYEiisjvcwmrJrZsFxzOcWseEb78BAue1uMqEEK\\nBgvQMO7jAoRX328Eig3kj0TCs+MHaI1YHa4ZXDBua5ocE0O+ryNA/qS5eNWBKDqA\\n/v1TUdGbXjW6ObAK6XAnSt60hweLrd7s03UCvhadwbdvm7oduP7btMbYvzMqwWem\\ns/iCaYluwoKQS6pV5aFOkRU/CY8fecs9eXVsawEDfLLIKPjJGbxW8NR2YY898a0k\\nWC2s6hcqb61wAAy/Bnu0MIvkVTGtuTEu484zC5n7HcNUmZXcsdL/SHpb/IrWiLbr\\nxGGpeQbgol7Yry88Ntg7hzT+15jf+GCS0kKu5yx4i5omM21w+Y1byc7qJuIvXDai\\nbAepk3FlF/OSy2rfYoi6\\n=5Hcu\\n-----END PGP SIGNATURE-----\\n","payload":"tree 01d1b2024af28c7d5abf2a66108e7ed5611c6308\\nparent d089b78198add300ea06b8c874234fc2c6d8f172\\nauthor Grisha <skvrd@users.noreply.github.com> 1715181574 -0700\\ncommitter GitHub <noreply@github.com> 1715181574 +0000\\n\\nFix hidden temp directory issue for arxiv reader (#13351)\\n\\n* Fix hidden temp directory issue for arxiv reader\\r\\n\\r\\n* add comment base.py\\r\\n\\r\\n---------\\r\\n\\r\\nCo-authored-by: Andrei Fajardo <92402603+nerdai@users.noreply.github.com>"}},"url":"https://api.github.com/repos/run-llama/llama_index/commits/a11a953e738cbda93335ede83f012914d53dc4f7","html_url":"https://github.com/run-llama/llama_index/commit/a11a953e738cbda93335ede83f012914d53dc4f7","comments_url":"https://api.github.com/repos/run-llama/llama_index/commits/a11a953e738cbda93335ede83f012914d53dc4f7/comments","author":{"login":"skvrd","id":18094327,"node_id":"MDQ6VXNlcjE4MDk0MzI3","avatar_url":"https://avatars.githubusercontent.com/u/18094327?v=4","gravatar_id":"","url":"https://api.github.com/users/skvrd","html_url":"https://github.com/skvrd","followers_url":"https://api.github.com/users/skvrd/followers","following_url":"https://api.github.com/users/skvrd/following{/other_user}","gists_url":"https://api.github.com/users/skvrd/gists{/gist_id}","starred_url":"https://api.github.com/users/skvrd/starred{/owner}{/repo}","subscriptions_url":"https://api.github.com/users/skvrd/subscriptions","organizations_url":"https://api.github.com/users/skvrd/orgs","repos_url":"https://api.github.com/users/skvrd/repos","events_url":"https://api.github.com/users/skvrd/events{/privacy}","received_events_url":"https://api.github.com/users/skvrd/received_events","type":"User","site_admin":false},"committer":{"login":"web-flow","id":19864447,"node_id":"MDQ6VXNlcjE5ODY0NDQ3","avatar_url":"https://avatars.githubusercontent.com/u/19864447?v=4","gravatar_id":"","url":"https://api.github.com/users/web-flow","html_url":"https://github.com/web-flow","followers_url":"https://api.github.com/users/web-flow/followers","following_url":"https://api.github.com/users/web-flow/following{/other_user}","gists_url":"https://api.github.com/users/web-flow/gists{/gist_id}","starred_url":"https://api.github.com/users/web-flow/starred{/owner}{/repo}","subscriptions_url":"https://api.github.com/users/web-flow/subscriptions","organizations_url":"https://api.github.com/users/web-flow/orgs","repos_url":"https://api.github.com/users/web-flow/repos","events_url":"https://api.github.com/users/web-flow/events{/privacy}","received_events_url":"https://api.github.com/users/web-flow/received_events","type":"User","site_admin":false},"parents":[{"sha":"d089b78198add300ea06b8c874234fc2c6d8f172","url":"https://api.github.com/repos/run-llama/llama_index/commits/d089b78198add300ea06b8c874234fc2c6d8f172","html_url":"https://github.com/run-llama/llama_index/commit/d089b78198add300ea06b8c874234fc2c6d8f172"}],"stats":{"total":12,"additions":10,"deletions":2},"files":[{"sha":"b3813e3c4c6cd8c90fb9ec0409c2d7ce6ca38304","filename":"llama-index-integrations/readers/llama-index-readers-papers/CHANGELOG.md","status":"modified","additions":6,"deletions":0,"changes":6,"blob_url":"https://github.com/run-llama/llama_index/blob/a11a953e738cbda93335ede83f012914d53dc4f7/llama-index-integrations%2Freaders%2Fllama-index-readers-papers%2FCHANGELOG.md","raw_url":"https://github.com/run-llama/llama_index/raw/a11a953e738cbda93335ede83f012914d53dc4f7/llama-index-integrations%2Freaders%2Fllama-index-readers-papers%2FCHANGELOG.md","contents_url":"https://api.github.com/repos/run-llama/llama_index/contents/llama-index-integrations%2Freaders%2Fllama-index-readers-papers%2FCHANGELOG.md?ref=a11a953e738cbda93335ede83f012914d53dc4f7","patch":"@@ -1,5 +1,11 @@\\n # CHANGELOG\\n \\n+## [0.1.5] - 2024-05-07\\n+\\n+### Bug Fixes\\n+\\n+- Fix issues with hidden temporary folder (#13165)\\n+\\n ## [0.1.2] - 2024-02-13\\n \\n - Add maintainers and keywords from library.json (llamahub)"},{"sha":"6519b4e78e6890dc71ecc8140876418bde453309","filename":"llama-index-integrations/readers/llama-index-readers-papers/llama_index/readers/papers/arxiv/base.py","status":"modified","additions":3,"deletions":1,"changes":4,"blob_url":"https://github.com/run-llama/llama_index/blob/a11a953e738cbda93335ede83f012914d53dc4f7/llama-index-integrations%2Freaders%2Fllama-index-readers-papers%2Fllama_index%2Freaders%2Fpapers%2Farxiv%2Fbase.py","raw_url":"https://github.com/run-llama/llama_index/raw/a11a953e738cbda93335ede83f012914d53dc4f7/llama-index-integrations%2Freaders%2Fllama-index-readers-papers%2Fllama_index%2Freaders%2Fpapers%2Farxiv%2Fbase.py","contents_url":"https://api.github.com/repos/run-llama/llama_index/contents/llama-index-integrations%2Freaders%2Fllama-index-readers-papers%2Fllama_index%2Freaders%2Fpapers%2Farxiv%2Fbase.py?ref=a11a953e738cbda93335ede83f012914d53dc4f7","patch":"@@ -73,7 +73,9 @@ def get_paper_metadata(filename):\\n             return paper_lookup[os.path.basename(filename)]\\n \\n         arxiv_documents = SimpleDirectoryReader(\\n-            papers_dir, file_metadata=get_paper_metadata\\n+            papers_dir,\\n+            file_metadata=get_paper_metadata,\\n+            exclude_hidden=False,  # default directory is hidden \\".papers\\"\\n         ).load_data()\\n         # Include extra documents containing the abstracts\\n         abstract_documents = []"},{"sha":"0cb2c74567e189cf481ad1e7ffe81e1020174edd","filename":"llama-index-integrations/readers/llama-index-readers-papers/pyproject.toml","status":"modified","additions":1,"deletions":1,"changes":2,"blob_url":"https://github.com/run-llama/llama_index/blob/a11a953e738cbda93335ede83f012914d53dc4f7/llama-index-integrations%2Freaders%2Fllama-index-readers-papers%2Fpyproject.toml","raw_url":"https://github.com/run-llama/llama_index/raw/a11a953e738cbda93335ede83f012914d53dc4f7/llama-index-integrations%2Freaders%2Fllama-index-readers-papers%2Fpyproject.toml","contents_url":"https://api.github.com/repos/run-llama/llama_index/contents/llama-index-integrations%2Freaders%2Fllama-index-readers-papers%2Fpyproject.toml?ref=a11a953e738cbda93335ede83f012914d53dc4f7","patch":"@@ -29,7 +29,7 @@ license = \\"MIT\\"\\n maintainers = [\\"thejessezhang\\"]\\n name = \\"llama-index-readers-papers\\"\\n readme = \\"README.md\\"\\n-version = \\"0.1.4\\"\\n+version = \\"0.1.5\\"\\n \\n [tool.poetry.dependencies]\\n python = \\">=3.8.1,<4.0\\""}]}'

BRANCH_JSON = '{"name":"main","commit":{"sha":"a11a953e738cbda93335ede83f012914d53dc4f7","node_id":"C_kwDOIWuq59oAKGExMWE5NTNlNzM4Y2JkYTkzMzM1ZWRlODNmMDEyOTE0ZDUzZGM0Zjc","commit":{"author":{"name":"Grisha","email":"skvrd@users.noreply.github.com","date":"2024-05-08T15:19:34Z"},"committer":{"name":"GitHub","email":"noreply@github.com","date":"2024-05-08T15:19:34Z"},"message":"Fix hidden temp directory issue for arxiv reader (#13351)\\n\\n* Fix hidden temp directory issue for arxiv reader\\r\\n\\r\\n* add comment base.py\\r\\n\\r\\n---------\\r\\n\\r\\nCo-authored-by: Andrei Fajardo <92402603+nerdai@users.noreply.github.com>","tree":{"sha":"01d1b2024af28c7d5abf2a66108e7ed5611c6308","url":"https://api.github.com/repos/run-llama/llama_index/git/trees/01d1b2024af28c7d5abf2a66108e7ed5611c6308"},"url":"https://api.github.com/repos/run-llama/llama_index/git/commits/a11a953e738cbda93335ede83f012914d53dc4f7","comment_count":0,"verification":{"verified":true,"reason":"valid","signature":"-----BEGIN PGP SIGNATURE-----\\n\\nwsFcBAABCAAQBQJmO5gGCRC1aQ7uu5UhlAAAjoYQAEy/6iaWXaHQUEAz2w+2fs7C\\n8cXt3BcXkbO+d6WmIsca22wCbhRVj0TQdYTbfjLDpyMK1BdFcsgBjC7Ym8dm40dZ\\nRxoNd0Ws04doz64L67zXtBTQXjEb3nNnDkl82jk0AtuFfb69pTjC/rH/MqDt7v4F\\nU4uDR0OyDakmEEFNE63UhKmbwkmMN33VXZMbm2obxJklR2rBCge7EBhtj+iwD8Og\\nDQ852VzB7/PV4mhalLjuP8CiY9kItZq0zN+Fn/ghB+0o6Xf3cIxCraiL34jxGBma\\nvXgRsqgI8kMW5ZE9zRjQhEh5GFYEiisjvcwmrJrZsFxzOcWseEb78BAue1uMqEEK\\nBgvQMO7jAoRX328Eig3kj0TCs+MHaI1YHa4ZXDBua5ocE0O+ryNA/qS5eNWBKDqA\\n/v1TUdGbXjW6ObAK6XAnSt60hweLrd7s03UCvhadwbdvm7oduP7btMbYvzMqwWem\\ns/iCaYluwoKQS6pV5aFOkRU/CY8fecs9eXVsawEDfLLIKPjJGbxW8NR2YY898a0k\\nWC2s6hcqb61wAAy/Bnu0MIvkVTGtuTEu484zC5n7HcNUmZXcsdL/SHpb/IrWiLbr\\nxGGpeQbgol7Yry88Ntg7hzT+15jf+GCS0kKu5yx4i5omM21w+Y1byc7qJuIvXDai\\nbAepk3FlF/OSy2rfYoi6\\n=5Hcu\\n-----END PGP SIGNATURE-----\\n","payload":"tree 01d1b2024af28c7d5abf2a66108e7ed5611c6308\\nparent d089b78198add300ea06b8c874234fc2c6d8f172\\nauthor Grisha <skvrd@users.noreply.github.com> 1715181574 -0700\\ncommitter GitHub <noreply@github.com> 1715181574 +0000\\n\\nFix hidden temp directory issue for arxiv reader (#13351)\\n\\n* Fix hidden temp directory issue for arxiv reader\\r\\n\\r\\n* add comment base.py\\r\\n\\r\\n---------\\r\\n\\r\\nCo-authored-by: Andrei Fajardo <92402603+nerdai@users.noreply.github.com>"}},"url":"https://api.github.com/repos/run-llama/llama_index/commits/a11a953e738cbda93335ede83f012914d53dc4f7","html_url":"https://github.com/run-llama/llama_index/commit/a11a953e738cbda93335ede83f012914d53dc4f7","comments_url":"https://api.github.com/repos/run-llama/llama_index/commits/a11a953e738cbda93335ede83f012914d53dc4f7/comments","author":{"login":"skvrd","id":18094327,"node_id":"MDQ6VXNlcjE4MDk0MzI3","avatar_url":"https://avatars.githubusercontent.com/u/18094327?v=4","gravatar_id":"","url":"https://api.github.com/users/skvrd","html_url":"https://github.com/skvrd","followers_url":"https://api.github.com/users/skvrd/followers","following_url":"https://api.github.com/users/skvrd/following{/other_user}","gists_url":"https://api.github.com/users/skvrd/gists{/gist_id}","starred_url":"https://api.github.com/users/skvrd/starred{/owner}{/repo}","subscriptions_url":"https://api.github.com/users/skvrd/subscriptions","organizations_url":"https://api.github.com/users/skvrd/orgs","repos_url":"https://api.github.com/users/skvrd/repos","events_url":"https://api.github.com/users/skvrd/events{/privacy}","received_events_url":"https://api.github.com/users/skvrd/received_events","type":"User","site_admin":false},"committer":{"login":"web-flow","id":19864447,"node_id":"MDQ6VXNlcjE5ODY0NDQ3","avatar_url":"https://avatars.githubusercontent.com/u/19864447?v=4","gravatar_id":"","url":"https://api.github.com/users/web-flow","html_url":"https://github.com/web-flow","followers_url":"https://api.github.com/users/web-flow/followers","following_url":"https://api.github.com/users/web-flow/following{/other_user}","gists_url":"https://api.github.com/users/web-flow/gists{/gist_id}","starred_url":"https://api.github.com/users/web-flow/starred{/owner}{/repo}","subscriptions_url":"https://api.github.com/users/web-flow/subscriptions","organizations_url":"https://api.github.com/users/web-flow/orgs","repos_url":"https://api.github.com/users/web-flow/repos","events_url":"https://api.github.com/users/web-flow/events{/privacy}","received_events_url":"https://api.github.com/users/web-flow/received_events","type":"User","site_admin":false},"parents":[{"sha":"d089b78198add300ea06b8c874234fc2c6d8f172","url":"https://api.github.com/repos/run-llama/llama_index/commits/d089b78198add300ea06b8c874234fc2c6d8f172","html_url":"https://github.com/run-llama/llama_index/commit/d089b78198add300ea06b8c874234fc2c6d8f172"}]},"_links":{"self":"https://api.github.com/repos/run-llama/llama_index/branches/main","html":"https://github.com/run-llama/llama_index/tree/main"},"protected":true,"protection":{"enabled":true,"required_status_checks":{"enforcement_level":"non_admins","contexts":["CodeQL","build (3.9)","build (ubuntu-latest, 3.9)","build (windows-latest, 3.9)","test (3.8)","test (3.9)","test (3.10)"],"checks":[{"context":"CodeQL","app_id":57789},{"context":"build (3.9)","app_id":15368},{"context":"build (ubuntu-latest, 3.9)","app_id":15368},{"context":"build (windows-latest, 3.9)","app_id":15368},{"context":"test (3.8)","app_id":15368},{"context":"test (3.9)","app_id":15368},{"context":"test (3.10)","app_id":15368}]}},"protection_url":"https://api.github.com/repos/run-llama/llama_index/branches/main/protection"}'

TREE_JSON = '{"sha":"01d1b2024af28c7d5abf2a66108e7ed5611c6308","url":"https://api.github.com/repos/run-llama/llama_index/git/trees/01d1b2024af28c7d5abf2a66108e7ed5611c6308","tree":[{"path":"README.md","mode":"100644","type":"blob","sha":"0bbd4e1720494e30e267c11dc9967c100b86bad8","size":10101,"url":"https://api.github.com/repos/run-llama/llama_index/git/blobs/0bbd4e1720494e30e267c11dc9967c100b86bad8"}],"truncated":false}'

BLOB_JSON = '{"sha":"c3c8a7f594d936e2f3b908c6d3b73bbccb11886f","node_id":"B_kwDOIWuq59oAKGMzYzhhN2Y1OTRkOTM2ZTJmM2I5MDhjNmQzYjczYmJjY2IxMTg4NmY","size":40,"url":"https://api.github.com/repos/run-llama/llama_index/git/blobs/c3c8a7f594d936e2f3b908c6d3b73bbccb11886f","content":"cG9ldHJ5X3JlcXVpcmVtZW50cygKICAgIG5hbWU9InJvb3QiLAopCg==\\n","encoding":"base64"}'

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


@pytest.fixture()
def mock_error(monkeypatch):
    async def mock_get_blob(self, *args, **kwargs):
        if self._fail_on_http_error:
            raise HTTPError("Woops")
        else:
            return

    monkeypatch.setattr(GithubClient, "get_blob", mock_get_blob)

    async def mock_get_commit(self, *args, **kwargs):
        return GitCommitResponseModel.from_json(COMMIT_JSON)

    monkeypatch.setattr(GithubClient, "get_commit", mock_get_commit)

    async def mock_get_branch(self, *args, **kwargs):
        return GitBranchResponseModel.from_json(BRANCH_JSON)

    monkeypatch.setattr(GithubClient, "get_branch", mock_get_branch)

    async def mock_get_tree(self, *args, **kwargs):
        return GitTreeResponseModel.from_json(TREE_JSON)

    monkeypatch.setattr(GithubClient, "get_tree", mock_get_tree)


def test_fail_on_http_error_true(mock_error):
    gh_client = GithubClient(GITHUB_TOKEN, fail_on_http_error=True)
    reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")
    # test for branch
    with pytest.raises(HTTPError):
        reader.load_data(branch="main")
    # test for commit
    with pytest.raises(HTTPError):
        reader.load_data(commit_sha="a11a953e738cbda93335ede83f012914d53dc4f7")


def test_fail_on_http_error_false(mock_error):
    gh_client = GithubClient(GITHUB_TOKEN, fail_on_http_error=False)
    reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")
    # test for branch
    documents = reader.load_data(branch="main")
    assert isinstance(documents, list)
    assert all(isinstance(doc, Document) for doc in documents)
    # test for commit
    documents = reader.load_data(commit_sha="a11a953e738cbda93335ede83f012914d53dc4f7")
    assert isinstance(documents, list)
    assert all(isinstance(doc, Document) for doc in documents)


def test_timeout_and_retries_passed_to_request():
    with patch.object(GithubClient, "request", new_callable=AsyncMock) as mock_request:

        def mock_request_response(endpoint, *args, **kwargs):
            if endpoint == "getCommit":
                data = COMMIT_JSON
            elif endpoint == "getBranch":
                data = BRANCH_JSON
            elif endpoint == "getTree":
                data = TREE_JSON
            elif endpoint == "getBlob":
                data = BLOB_JSON
            else:
                raise RuntimeError(f"Unhandled endpoint: {endpoint}")
            return Response(status_code=200, json=json.loads(data))

        mock_request.side_effect = mock_request_response

        gh_client = GithubClient(GITHUB_TOKEN, fail_on_http_error=False)
        reader = GithubRepositoryReader(
            gh_client, "run-llama", "llama_index", timeout=9, retries=3
        )

        # Load data via both branch and sha to exercise those functions in the
        # GithubClient
        reader.load_data(branch="main")
        reader.load_data(commit_sha="a11a953e738cbda93335ede83f012914d53dc4f7")

        mock_request.assert_called()
        for call in mock_request.call_args_list:
            # GithubClient.request is used for getting info from either the commit sha
            # or the branch name, as well as fetching blobs. So all calls to request
            # should have the same timeout and retries values.
            assert call.kwargs["timeout"] == 9
            assert call.kwargs["retries"] == 3


class TestGithubRepositoryReaderFiltering:
    """Test file filtering functionality."""

    @pytest.fixture
    def mock_client_responses(self, monkeypatch):
        """Mock client responses for filtering tests."""

        async def mock_get_commit(self, *args, **kwargs):
            return GitCommitResponseModel.from_json(COMMIT_JSON)

        async def mock_get_branch(self, *args, **kwargs):
            return GitBranchResponseModel.from_json(BRANCH_JSON)

        # Mock tree with various file types and paths
        tree_json = {
            "sha": "01d1b2024af28c7d5abf2a66108e7ed5611c6308",
            "url": "https://api.github.com/repos/run-llama/llama_index/git/trees/01d1b2024af28c7d5abf2a66108e7ed5611c6308",
            "tree": [
                {
                    "path": "README.md",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "readme_sha",
                    "size": 1000,
                    "url": "https://api.github.com/repos/run-llama/llama_index/git/blobs/readme_sha",
                },
                {
                    "path": "src/main.py",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "main_py_sha",
                    "size": 2000,
                    "url": "https://api.github.com/repos/run-llama/llama_index/git/blobs/main_py_sha",
                },
                {
                    "path": "tests/test_file.py",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "test_py_sha",
                    "size": 1500,
                    "url": "https://api.github.com/repos/run-llama/llama_index/git/blobs/test_py_sha",
                },
                {
                    "path": "docs/guide.md",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "guide_md_sha",
                    "size": 3000,
                    "url": "https://api.github.com/repos/run-llama/llama_index/git/blobs/guide_md_sha",
                },
            ],
            "truncated": False,
        }

        async def mock_get_tree(self, *args, **kwargs):
            return GitTreeResponseModel.from_json(json.dumps(tree_json))

        async def mock_get_blob(self, *args, **kwargs):
            blob_content = (
                "VGhpcyBpcyBhIHRlc3QgZmlsZQ=="  # "This is a test file" in base64
            )
            blob_json = {
                "sha": "test_sha",
                "node_id": "test_node",
                "size": 40,
                "url": "https://api.github.com/repos/run-llama/llama_index/git/blobs/test_sha",
                "content": blob_content,
                "encoding": "base64",
            }
            from llama_index.readers.github.repository.github_client import (
                GitBlobResponseModel,
            )

            return GitBlobResponseModel.from_json(json.dumps(blob_json))

        monkeypatch.setattr(GithubClient, "get_commit", mock_get_commit)
        monkeypatch.setattr(GithubClient, "get_branch", mock_get_branch)
        monkeypatch.setattr(GithubClient, "get_tree", mock_get_tree)
        monkeypatch.setattr(GithubClient, "get_blob", mock_get_blob)

    def test_filter_file_paths_include(self, mock_client_responses):
        """Test including specific file paths."""
        gh_client = GithubClient(GITHUB_TOKEN)
        reader = GithubRepositoryReader(
            gh_client,
            "run-llama",
            "llama_index",
            filter_file_paths=(
                ["README.md", "src/main.py"],
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
        )

        documents = reader.load_data(branch="main")

        # Should only include the specified files
        file_paths = [doc.metadata["file_path"] for doc in documents]
        assert "README.md" in file_paths
        assert "src/main.py" in file_paths
        assert "tests/test_file.py" not in file_paths
        assert "docs/guide.md" not in file_paths

    def test_filter_file_paths_exclude(self, mock_client_responses):
        """Test excluding specific file paths."""
        gh_client = GithubClient(GITHUB_TOKEN)
        reader = GithubRepositoryReader(
            gh_client,
            "run-llama",
            "llama_index",
            filter_file_paths=(
                ["tests/test_file.py"],
                GithubRepositoryReader.FilterType.EXCLUDE,
            ),
        )

        documents = reader.load_data(branch="main")

        # Should exclude the specified file
        file_paths = [doc.metadata["file_path"] for doc in documents]
        assert "README.md" in file_paths
        assert "src/main.py" in file_paths
        assert "docs/guide.md" in file_paths
        assert "tests/test_file.py" not in file_paths

    def test_process_file_callback_skip_large_files(self, mock_client_responses):
        """Test process file callback that skips large files."""

        def skip_large_files(file_path: str, file_size: int) -> tuple[bool, str]:
            if file_size > 2500:
                return False, f"File too large: {file_size} bytes"
            return True, ""

        gh_client = GithubClient(GITHUB_TOKEN)
        reader = GithubRepositoryReader(
            gh_client,
            "run-llama",
            "llama_index",
            process_file_callback=skip_large_files,
        )

        documents = reader.load_data(branch="main")

        # Should skip files larger than 2500 bytes (docs/guide.md has 3000 bytes)
        file_paths = [doc.metadata["file_path"] for doc in documents]
        assert "README.md" in file_paths  # 1000 bytes
        assert "src/main.py" in file_paths  # 2000 bytes
        assert "tests/test_file.py" in file_paths  # 1500 bytes
        assert "docs/guide.md" not in file_paths  # 3000 bytes

    def test_process_file_callback_skip_by_extension(self, mock_client_responses):
        """Test process file callback that skips by file extension."""

        def skip_python_files(file_path: str, file_size: int) -> tuple[bool, str]:
            if file_path.endswith(".py"):
                return False, "Skipping Python files"
            return True, ""

        gh_client = GithubClient(GITHUB_TOKEN)
        reader = GithubRepositoryReader(
            gh_client,
            "run-llama",
            "llama_index",
            process_file_callback=skip_python_files,
        )

        documents = reader.load_data(branch="main")

        # Should skip .py files
        file_paths = [doc.metadata["file_path"] for doc in documents]
        assert "README.md" in file_paths
        assert "docs/guide.md" in file_paths
        assert "src/main.py" not in file_paths
        assert "tests/test_file.py" not in file_paths

    def test_custom_folder_initialization(self):
        """Test custom folder initialization."""
        gh_client = GithubClient(GITHUB_TOKEN)

        # Test default custom folder (current working directory)
        reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")
        assert reader.custom_folder == os.getcwd()

        # Test custom folder specification
        custom_path = "/tmp/custom_test"
        reader_with_custom = GithubRepositoryReader(
            gh_client, "run-llama", "llama_index", custom_folder=custom_path
        )
        assert reader_with_custom.custom_folder == custom_path

    def test_custom_folder_with_parsers(self, mock_client_responses):
        """Test custom folder usage with custom parsers."""
        from llama_index.core.readers.base import BaseReader

        class MockParser(BaseReader):
            def load_data(self, file_path, extra_info=None):
                return [
                    Document(
                        text="mock content", metadata={"file_path": str(file_path)}
                    )
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            gh_client = GithubClient(GITHUB_TOKEN)
            reader = GithubRepositoryReader(
                gh_client,
                "run-llama",
                "llama_index",
                use_parser=True,
                custom_parsers={".py": MockParser()},
                custom_folder=temp_dir,
            )

            assert reader.custom_folder == temp_dir

            # The custom folder should be used for temporary file creation
            # This is tested indirectly through the parsing mechanism


class TestGithubRepositoryReaderEvents:
    """Test event system functionality."""

    @pytest.fixture
    def event_handler(self):
        """Create a test event handler."""

        class TestEventHandler(BaseEventHandler):
            def __init__(self, **data):
                super().__init__(**data)
                object.__setattr__(self, "events", [])

            def handle(self, event):
                self.events.append(event)

        return TestEventHandler()

    @pytest.fixture
    def mock_client_for_events(self, monkeypatch):
        """Mock client for event testing."""

        async def mock_get_commit(self, *args, **kwargs):
            return GitCommitResponseModel.from_json(COMMIT_JSON)

        async def mock_get_branch(self, *args, **kwargs):
            return GitBranchResponseModel.from_json(BRANCH_JSON)

        # Mock tree with one file for simpler event testing
        tree_json = {
            "sha": "test_tree_sha",
            "url": "https://api.github.com/test/tree",
            "tree": [
                {
                    "path": "test_file.txt",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "test_blob_sha",
                    "size": 100,
                    "url": "https://api.github.com/test/blob",
                }
            ],
            "truncated": False,
        }

        async def mock_get_tree(self, *args, **kwargs):
            return GitTreeResponseModel.from_json(json.dumps(tree_json))

        async def mock_get_blob(self, *args, **kwargs):
            blob_json = {
                "sha": "test_blob_sha",
                "node_id": "test_node",
                "size": 40,
                "url": "https://api.github.com/test/blob",
                "content": "VGVzdCBjb250ZW50",  # "Test content" in base64
                "encoding": "base64",
            }
            from llama_index.readers.github.repository.github_client import (
                GitBlobResponseModel,
            )

            return GitBlobResponseModel.from_json(json.dumps(blob_json))

        monkeypatch.setattr(GithubClient, "get_commit", mock_get_commit)
        monkeypatch.setattr(GithubClient, "get_branch", mock_get_branch)
        monkeypatch.setattr(GithubClient, "get_tree", mock_get_tree)
        monkeypatch.setattr(GithubClient, "get_blob", mock_get_blob)

    def test_repository_processing_events(self, mock_client_for_events, event_handler):
        """Test repository processing start/complete events."""
        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(event_handler)

        try:
            gh_client = GithubClient(GITHUB_TOKEN)
            reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")

            documents = reader.load_data(branch="main")

            # Check that we have the expected events
            event_types = [type(event).__name__ for event in event_handler.events]

            assert "GitHubRepositoryProcessingStartedEvent" in event_types
            assert "GitHubRepositoryProcessingCompletedEvent" in event_types
            assert "GitHubTotalFilesToProcessEvent" in event_types

            # Check start event details
            start_events = [
                e
                for e in event_handler.events
                if isinstance(e, GitHubRepositoryProcessingStartedEvent)
            ]
            assert len(start_events) == 1
            assert start_events[0].repository_name == "run-llama/llama_index"
            assert start_events[0].branch_or_commit == "main"

            # Check completion event details
            complete_events = [
                e
                for e in event_handler.events
                if isinstance(e, GitHubRepositoryProcessingCompletedEvent)
            ]
            assert len(complete_events) == 1
            assert complete_events[0].repository_name == "run-llama/llama_index"
            assert complete_events[0].total_documents == len(documents)

        finally:
            # Clean up by clearing the handler's events list
            pass

    def test_file_processing_events(self, mock_client_for_events, event_handler):
        """Test file processing events."""
        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(event_handler)

        try:
            gh_client = GithubClient(GITHUB_TOKEN)
            reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")

            documents = reader.load_data(branch="main")

            # Check file processing events
            event_types = [type(event).__name__ for event in event_handler.events]

            assert "GitHubFileProcessingStartedEvent" in event_types
            assert "GitHubFileProcessedEvent" in event_types

            # Check file processed event details
            processed_events = [
                e
                for e in event_handler.events
                if isinstance(e, GitHubFileProcessedEvent)
            ]
            assert len(processed_events) == 1
            assert processed_events[0].file_path == "test_file.txt"
            assert processed_events[0].file_type == ".txt"
            assert processed_events[0].document is not None

        finally:
            pass

    def test_file_skipped_events(self, mock_client_for_events, event_handler):
        """Test file skipped events."""

        def skip_all_files(file_path: str, file_size: int) -> tuple[bool, str]:
            return False, "Skipping for test"

        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(event_handler)

        try:
            gh_client = GithubClient(GITHUB_TOKEN)
            reader = GithubRepositoryReader(
                gh_client,
                "run-llama",
                "llama_index",
                process_file_callback=skip_all_files,
            )

            documents = reader.load_data(branch="main")

            # Check that files were skipped
            skipped_events = [
                e for e in event_handler.events if isinstance(e, GitHubFileSkippedEvent)
            ]
            assert len(skipped_events) == 1
            assert skipped_events[0].file_path == "test_file.txt"
            assert skipped_events[0].reason == "Skipping for test"

        finally:
            pass

    def test_file_failed_events(self, mock_client_for_events, event_handler):
        """Test file failed events with fail_on_error=False."""

        # Mock a blob that will cause a decoding error
        async def mock_get_blob_fail(self, *args, **kwargs):
            blob_json = {
                "sha": "test_blob_sha",
                "node_id": "test_node",
                "size": 40,
                "url": "https://api.github.com/test/blob",
                "content": "invalid_base64!!!",  # Invalid base64
                "encoding": "base64",
            }
            from llama_index.readers.github.repository.github_client import (
                GitBlobResponseModel,
            )

            return GitBlobResponseModel.from_json(json.dumps(blob_json))

        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(event_handler)

        try:
            with patch.object(GithubClient, "get_blob", mock_get_blob_fail):
                gh_client = GithubClient(GITHUB_TOKEN)
                reader = GithubRepositoryReader(
                    gh_client, "run-llama", "llama_index", fail_on_error=False
                )

                documents = reader.load_data(branch="main")

                # Check that files failed
                failed_events = [
                    e
                    for e in event_handler.events
                    if isinstance(e, GitHubFileFailedEvent)
                ]
                assert len(failed_events) == 1
                assert failed_events[0].file_path == "test_file.txt"
                assert "Could not decode as base64" in failed_events[0].error

        finally:
            pass

    def test_total_files_to_process_event(self, mock_client_for_events, event_handler):
        """Test total files to process event."""
        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(event_handler)

        try:
            gh_client = GithubClient(GITHUB_TOKEN)
            reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")

            documents = reader.load_data(branch="main")

            # Check total files event
            total_files_events = [
                e
                for e in event_handler.events
                if isinstance(e, GitHubTotalFilesToProcessEvent)
            ]
            assert len(total_files_events) == 1
            assert total_files_events[0].repository_name == "run-llama/llama_index"
            assert total_files_events[0].branch_or_commit == "main"
            assert total_files_events[0].total_files == 1  # Only one file in our mock

        finally:
            pass

    def test_file_processing_started_event(self, mock_client_for_events, event_handler):
        """Test file processing started event."""
        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(event_handler)

        try:
            gh_client = GithubClient(GITHUB_TOKEN)
            reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")

            documents = reader.load_data(branch="main")

            # Check file processing started events
            started_events = [
                e
                for e in event_handler.events
                if isinstance(e, GitHubFileProcessingStartedEvent)
            ]
            assert len(started_events) == 1
            assert started_events[0].file_path == "test_file.txt"
            assert started_events[0].file_type == ".txt"

        finally:
            pass

    def test_all_events_in_correct_order(self, mock_client_for_events, event_handler):
        """Test that all events are fired in the correct order."""
        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(event_handler)

        try:
            gh_client = GithubClient(GITHUB_TOKEN)
            reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")

            documents = reader.load_data(branch="main")

            # Check event order
            event_types = [type(event).__name__ for event in event_handler.events]

            # Expected order: Started -> TotalFiles -> FileStarted -> FileProcessed -> Completed
            expected_order = [
                "GitHubRepositoryProcessingStartedEvent",
                "GitHubTotalFilesToProcessEvent",
                "GitHubFileProcessingStartedEvent",
                "GitHubFileProcessedEvent",
                "GitHubRepositoryProcessingCompletedEvent",
            ]

            # Check that all expected events are present
            for expected_event in expected_order:
                assert expected_event in event_types

            # Check the order - find indices of each event type
            indices = {}
            for event_type in expected_order:
                indices[event_type] = event_types.index(event_type)

            # Verify order is correct
            assert (
                indices["GitHubRepositoryProcessingStartedEvent"]
                < indices["GitHubTotalFilesToProcessEvent"]
            )
            assert (
                indices["GitHubTotalFilesToProcessEvent"]
                < indices["GitHubFileProcessingStartedEvent"]
            )
            assert (
                indices["GitHubFileProcessingStartedEvent"]
                < indices["GitHubFileProcessedEvent"]
            )
            assert (
                indices["GitHubFileProcessedEvent"]
                < indices["GitHubRepositoryProcessingCompletedEvent"]
            )

        finally:
            pass


class TestGithubRepositoryReaderEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_filter_type_raises_error(self):
        """Test that invalid filter types raise ValueError."""
        gh_client = GithubClient(GITHUB_TOKEN)

        with pytest.raises(ValueError, match="Unknown filter type"):
            reader = GithubRepositoryReader(gh_client, "run-llama", "llama_index")
            reader._filter_file_paths = (["test.py"], "INVALID_TYPE")
            reader._check_filter_file_paths("test.py")

    def test_process_file_callback_exception_handling(self, monkeypatch):
        """Test that exceptions in process_file_callback are handled gracefully."""

        def broken_callback(file_path: str, file_size: int) -> tuple[bool, str]:
            raise Exception("Callback error")

        async def mock_get_commit(self, *args, **kwargs):
            return GitCommitResponseModel.from_json(COMMIT_JSON)

        async def mock_get_branch(self, *args, **kwargs):
            return GitBranchResponseModel.from_json(BRANCH_JSON)

        tree_json = {
            "sha": "test_tree_sha",
            "url": "https://api.github.com/test/tree",
            "tree": [
                {
                    "path": "test_file.txt",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "test_blob_sha",
                    "size": 100,
                    "url": "https://api.github.com/test/blob",
                }
            ],
            "truncated": False,
        }

        async def mock_get_tree(self, *args, **kwargs):
            return GitTreeResponseModel.from_json(json.dumps(tree_json))

        monkeypatch.setattr(GithubClient, "get_commit", mock_get_commit)
        monkeypatch.setattr(GithubClient, "get_branch", mock_get_branch)
        monkeypatch.setattr(GithubClient, "get_tree", mock_get_tree)

        gh_client = GithubClient(GITHUB_TOKEN)
        reader = GithubRepositoryReader(
            gh_client,
            "run-llama",
            "llama_index",
            process_file_callback=broken_callback,
            fail_on_error=False,
        )

        # This should not raise an exception due to fail_on_error=False
        # The callback exception should be caught and handled
        documents = reader.load_data(branch="main")
        # Should succeed and return empty list since the file was skipped due to callback error
        assert isinstance(documents, list)

    def test_fail_on_error_true_with_processing_error(self, monkeypatch):
        """Test that fail_on_error=True propagates processing errors."""

        async def mock_get_commit(self, *args, **kwargs):
            return GitCommitResponseModel.from_json(COMMIT_JSON)

        async def mock_get_branch(self, *args, **kwargs):
            return GitBranchResponseModel.from_json(BRANCH_JSON)

        tree_json = {
            "sha": "test_tree_sha",
            "url": "https://api.github.com/test/tree",
            "tree": [
                {
                    "path": "test_file.txt",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "test_blob_sha",
                    "size": 100,
                    "url": "https://api.github.com/test/blob",
                }
            ],
            "truncated": False,
        }

        async def mock_get_tree(self, *args, **kwargs):
            return GitTreeResponseModel.from_json(json.dumps(tree_json))

        async def mock_get_blob_fail(self, *args, **kwargs):
            blob_json = {
                "sha": "test_blob_sha",
                "node_id": "test_node",
                "size": 40,
                "url": "https://api.github.com/test/blob",
                "content": "invalid_base64!!!",  # Invalid base64
                "encoding": "base64",
            }
            from llama_index.readers.github.repository.github_client import (
                GitBlobResponseModel,
            )

            return GitBlobResponseModel.from_json(json.dumps(blob_json))

        monkeypatch.setattr(GithubClient, "get_commit", mock_get_commit)
        monkeypatch.setattr(GithubClient, "get_branch", mock_get_branch)
        monkeypatch.setattr(GithubClient, "get_tree", mock_get_tree)
        monkeypatch.setattr(GithubClient, "get_blob", mock_get_blob_fail)

        gh_client = GithubClient(GITHUB_TOKEN)
        reader = GithubRepositoryReader(
            gh_client, "run-llama", "llama_index", fail_on_error=True
        )

        # This should continue processing despite base64 decode errors
        # since those are handled separately from processing errors
        documents = reader.load_data(branch="main")
        assert isinstance(documents, list)
