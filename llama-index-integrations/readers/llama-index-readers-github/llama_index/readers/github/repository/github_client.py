"""
Github API client for the GPT-Index library.

This module contains the Github API client for the GPT-Index library.
It is used by the Github readers to retrieve the data from Github.
"""

import os

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from dataclasses_json import DataClassJsonMixin, config
from httpx import HTTPError


@dataclass
class GitTreeResponseModel(DataClassJsonMixin):
    """
    Dataclass for the response from the Github API's getTree endpoint.

    Attributes:
        - sha (str): SHA1 checksum ID of the tree.
        - url (str): URL for the tree.
        - tree (List[GitTreeObject]): List of objects in the tree.
        - truncated (bool): Whether the tree is truncated.

    Examples:
        >>> tree = client.get_tree("owner", "repo", "branch")
        >>> tree.sha

    """

    @dataclass
    class GitTreeObject(DataClassJsonMixin):
        """
        Dataclass for the objects in the tree.

        Attributes:
            - path (str): Path to the object.
            - mode (str): Mode of the object.
            - type (str): Type of the object.
            - sha (str): SHA1 checksum ID of the object.
            - url (str): URL for the object.
            - size (Optional[int]): Size of the object (only for blobs).

        """

        path: str
        mode: str
        type: str
        sha: str
        url: str
        size: Optional[int] = None

    sha: str
    url: str
    tree: List[GitTreeObject]
    truncated: bool


@dataclass
class GitBlobResponseModel(DataClassJsonMixin):
    """
    Dataclass for the response from the Github API's getBlob endpoint.

    Attributes:
        - content (str): Content of the blob.
        - encoding (str): Encoding of the blob.
        - url (str): URL for the blob.
        - sha (str): SHA1 checksum ID of the blob.
        - size (int): Size of the blob.
        - node_id (str): Node ID of the blob.

    """

    content: str
    encoding: str
    url: str
    sha: str
    size: int
    node_id: str


@dataclass
class GitCommitResponseModel(DataClassJsonMixin):
    """
    Dataclass for the response from the Github API's getCommit endpoint.

    Attributes:
        - tree (Tree): Tree object for the commit.

    """

    @dataclass
    class Commit(DataClassJsonMixin):
        """Dataclass for the commit object in the commit. (commit.commit)."""

        @dataclass
        class Tree(DataClassJsonMixin):
            """
            Dataclass for the tree object in the commit.

            Attributes:
                - sha (str): SHA for the commit

            """

            sha: str

        tree: Tree

    commit: Commit
    url: str
    sha: str


@dataclass
class GitBranchResponseModel(DataClassJsonMixin):
    """
    Dataclass for the response from the Github API's getBranch endpoint.

    Attributes:
        - commit (Commit): Commit object for the branch.

    """

    @dataclass
    class Commit(DataClassJsonMixin):
        """Dataclass for the commit object in the branch. (commit.commit)."""

        @dataclass
        class Commit(DataClassJsonMixin):
            """Dataclass for the commit object in the commit. (commit.commit.tree)."""

            @dataclass
            class Tree(DataClassJsonMixin):
                """
                Dataclass for the tree object in the commit.

                Usage: commit.commit.tree.sha
                """

                sha: str

            tree: Tree

        commit: Commit

    @dataclass
    class Links(DataClassJsonMixin):
        _self: str = field(metadata=config(field_name="self"))
        html: str

    commit: Commit
    name: str
    _links: Links


class BaseGithubClient(Protocol):
    def get_all_endpoints(self) -> Dict[str, str]:
        ...

    async def request(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Any:
        ...

    async def get_tree(
        self,
        owner: str,
        repo: str,
        tree_sha: str,
    ) -> GitTreeResponseModel:
        ...

    async def get_blob(
        self,
        owner: str,
        repo: str,
        file_sha: str,
    ) -> Optional[GitBlobResponseModel]:
        ...

    async def get_commit(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> GitCommitResponseModel:
        ...

    async def get_branch(
        self,
        owner: str,
        repo: str,
        branch: Optional[str],
        branch_name: Optional[str],
    ) -> GitBranchResponseModel:
        ...


class GithubClient:
    """
    An asynchronous client for interacting with the Github API.

    This client is used for making API requests to Github.
    It provides methods for accessing the Github API endpoints.
    The client requires a Github token for authentication,
    which can be passed as an argument or set as an environment variable.
    If no Github token is provided, the client will raise a ValueError.

    Examples:
        >>> client = GithubClient("my_github_token")
        >>> branch_info = client.get_branch("owner", "repo", "branch")

    """

    DEFAULT_BASE_URL = "https://api.github.com"
    DEFAULT_API_VERSION = "2022-11-28"

    def __init__(
        self,
        github_token: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
        verbose: bool = False,
        fail_on_http_error: bool = True,
    ) -> None:
        """
        Initialize the GithubClient.

        Args:
            - github_token (str): Github token for authentication.
                If not provided, the client will try to get it from
                the GITHUB_TOKEN environment variable.
            - base_url (str): Base URL for the Github API
                (defaults to "https://api.github.com").
            - api_version (str): Github API version (defaults to "2022-11-28").
            - verbose (bool): Whether to print verbose output (defaults to False).
            - fail_on_http_error (bool): Whether to raise an exception on HTTP errors (defaults to True).

        Raises:
            ValueError: If no Github token is provided.

        """
        if github_token is None:
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token is None:
                raise ValueError(
                    "Please provide a Github token. "
                    + "You can do so by passing it as an argument to the GithubReader,"
                    + "or by setting the GITHUB_TOKEN environment variable."
                )

        self._base_url = base_url
        self._api_version = api_version
        self._verbose = verbose
        self._fail_on_http_error = fail_on_http_error

        self._endpoints = {
            "getTree": "/repos/{owner}/{repo}/git/trees/{tree_sha}",
            "getBranch": "/repos/{owner}/{repo}/branches/{branch}",
            "getBlob": "/repos/{owner}/{repo}/git/blobs/{file_sha}",
            "getCommit": "/repos/{owner}/{repo}/commits/{commit_sha}",
        }

        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": f"{self._api_version}",
        }

    def get_all_endpoints(self) -> Dict[str, str]:
        """Get all available endpoints."""
        return {**self._endpoints}

    async def request(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, Any] = {},
        timeout: Optional[int] = 5,
        retries: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Make an API request to the Github API.

        This method is used for making API requests to the Github API.
        It is used internally by the other methods in the client.

        Args:
            - `endpoint (str)`: Name of the endpoint to make the request to.
            - `method (str)`: HTTP method to use for the request.
            - `headers (dict)`: HTTP headers to include in the request.
            - `timeout (int or None)`: Timeout for the request in seconds. Default is 5.
            - `retries (int)`: Number of retries for the request. Default is 0.
            - `**kwargs`: Keyword arguments to pass to the endpoint URL.

        Returns:
            - `response (httpx.Response)`: Response from the API request.

        Raises:
            - ImportError: If the `httpx` library is not installed.
            - httpx.HTTPError: If the API request fails and fail_on_http_error is True.

        Examples:
            >>> response = client.request("getTree", "GET",
                                owner="owner", repo="repo",
                                tree_sha="tree_sha", timeout=5, retries=0)

        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "Please install httpx to use the GithubRepositoryReader. "
                "You can do so by running `pip install httpx`."
            )

        _headers = {**self._headers, **headers}

        _client: httpx.AsyncClient
        async with httpx.AsyncClient(
            headers=_headers,
            base_url=self._base_url,
            timeout=timeout,
            transport=httpx.AsyncHTTPTransport(retries=retries),
        ) as _client:
            try:
                response = await _client.request(
                    method, url=self._endpoints[endpoint].format(**kwargs)
                )
            except httpx.HTTPError as excp:
                print(f"HTTP Exception for {excp.request.url} - {excp}")
                raise excp  # noqa: TRY201
            return response

    async def get_branch(
        self,
        owner: str,
        repo: str,
        branch: Optional[str] = None,
        branch_name: Optional[str] = None,
        timeout: Optional[int] = 5,
        retries: int = 0,
    ) -> GitBranchResponseModel:
        """
        Get information about a branch. (Github API endpoint: getBranch).

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `branch (str)`: Name of the branch.
            - `branch_name (str)`: Name of the branch (alternative to `branch`).
            - `timeout (int or None)`: Timeout for the request in seconds. Default is 5.
            - `retries (int)`: Number of retries for the request. Default is 0.

        Returns:
            - `branch_info (GitBranchResponseModel)`: Information about the branch.

        Examples:
            >>> branch_info = client.get_branch("owner", "repo", "branch")

        """
        if branch is None:
            if branch_name is None:
                raise ValueError("Either branch or branch_name must be provided.")
            branch = branch_name

        return GitBranchResponseModel.from_json(
            (
                await self.request(
                    "getBranch",
                    "GET",
                    owner=owner,
                    repo=repo,
                    branch=branch,
                    timeout=timeout,
                    retries=retries,
                )
            ).text
        )

    async def get_tree(
        self,
        owner: str,
        repo: str,
        tree_sha: str,
        timeout: Optional[int] = 5,
        retries: int = 0,
    ) -> GitTreeResponseModel:
        """
        Get information about a tree. (Github API endpoint: getTree).

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `tree_sha (str)`: SHA of the tree.
            - `timeout (int or None)`: Timeout for the request in seconds. Default is 5.
            - `retries (int)`: Number of retries for the request. Default is 0.

        Returns:
            - `tree_info (GitTreeResponseModel)`: Information about the tree.

        Examples:
            >>> tree_info = client.get_tree("owner", "repo", "tree_sha")

        """
        return GitTreeResponseModel.from_json(
            (
                await self.request(
                    "getTree",
                    "GET",
                    owner=owner,
                    repo=repo,
                    tree_sha=tree_sha,
                    timeout=timeout,
                    retries=retries,
                )
            ).text
        )

    async def get_blob(
        self,
        owner: str,
        repo: str,
        file_sha: str,
        timeout: Optional[int] = 5,
        retries: int = 0,
    ) -> Optional[GitBlobResponseModel]:
        """
        Get information about a blob. (Github API endpoint: getBlob).

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `file_sha (str)`: SHA of the file.
            - `timeout (int or None)`: Timeout for the request in seconds. Default is 5.
            - `retries (int)`: Number of retries for the request. Default is 0.

        Returns:
            - `blob_info (GitBlobResponseModel)`: Information about the blob.

        Examples:
            >>> blob_info = client.get_blob("owner", "repo", "file_sha")

        """
        try:
            return GitBlobResponseModel.from_json(
                (
                    await self.request(
                        "getBlob",
                        "GET",
                        owner=owner,
                        repo=repo,
                        file_sha=file_sha,
                        timeout=timeout,
                        retries=retries,
                    )
                ).text
            )
        except KeyError:
            print(f"Failed to get blob for {owner}/{repo}/{file_sha}")
            return None
        except HTTPError as excp:
            print(f"HTTP Exception for {excp.request.url} - {excp}")
            if self._fail_on_http_error:
                raise
            else:
                return None

    async def get_commit(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        timeout: Optional[int] = 5,
        retries: int = 0,
    ) -> GitCommitResponseModel:
        """
        Get information about a commit. (Github API endpoint: getCommit).

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `commit_sha (str)`: SHA of the commit.
            - `timeout (int or None)`: Timeout for the request in seconds. Default is 5.
            - `retries (int)`: Number of retries for the request. Default is 0.

        Returns:
            - `commit_info (GitCommitResponseModel)`: Information about the commit.

        Examples:
            >>> commit_info = client.get_commit("owner", "repo", "commit_sha")

        """
        return GitCommitResponseModel.from_json(
            (
                await self.request(
                    "getCommit",
                    "GET",
                    owner=owner,
                    repo=repo,
                    commit_sha=commit_sha,
                    timeout=timeout,
                    retries=retries,
                )
            ).text
        )


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Test the GithubClient."""
        client = GithubClient()
        response = await client.get_tree(
            owner="ahmetkca", repo="CommitAI", tree_sha="with-body"
        )

        for obj in response.tree:
            if obj.type == "blob":
                print(obj.path)
                print(obj.sha)
                blob_response = await client.get_blob(
                    owner="ahmetkca", repo="CommitAI", file_sha=obj.sha
                )
                print(blob_response.content if blob_response else "None")

    asyncio.run(main())
