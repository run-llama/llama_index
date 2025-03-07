from __future__ import annotations

from llama_index.readers.github import GithubRepositoryReader
from llama_index.readers.github.repository.github_client import (
    GithubClient,
    GitBranchResponseModel,
    GitTreeResponseModel,
    GitCommitResponseModel,
)
from llama_index.core.schema import Document

import json
import os
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
