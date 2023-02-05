import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dataclasses_json import DataClassJsonMixin


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
    class Tree(DataClassJsonMixin):
        """
        Dataclass for the tree object in the commit.

        Attributes:
            - sha (str): SHA for the commit
        """

        sha: str

    tree: Tree


@dataclass
class GitBranchResponseModel(DataClassJsonMixin):
    """
    Dataclass for the response from the Github API's getBranch endpoint.

    Attributes:
        - commit (Commit): Commit object for the branch.
    """

    @dataclass
    class Commit(DataClassJsonMixin):
        @dataclass
        class Commit(DataClassJsonMixin):
            @dataclass
            class Tree(DataClassJsonMixin):
                sha: str

            tree: Tree

        commit: Commit

    commit: Commit


class GithubClient:
    DEFAULT_BASE_URL = "https://api.github.com"
    DEFAULT_API_VERSION = "2022-11-28"

    def __init__(
        self,
        github_token: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
    ) -> None:
        if github_token is None:
            self.github_token = os.getenv("GITHUB_TOKEN")
            if self.github_token is None:
                raise ValueError(
                    "Please provide a Github token. You can do so by passing it as an argument to the GithubReader, or by setting the GITHUB_TOKEN environment variable."
                )
        else:
            self.github_token = github_token
        self.base_url = base_url
        self.api_version = api_version

        self.__endpoints = {
            "getTree": "/repos/{owner}/{repo}/git/trees/{tree_sha}",
            "getBranch": "/repos/{owner}/{repo}/branches/{branch}",
            "getBlob": "/repos/{owner}/{repo}/git/blobs/{file_sha}",
            "getCommit": "/repos/{owner}/{repo}/commits/{commit_sha}",
        }

        self.__headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.github_token}",
            "X-GitHub-Api-Version": f"{self.api_version}",
        }

    def get_all_endpoints(self) -> Dict[str, str]:
        return {**self.__endpoints}

    async def request(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, Any] = {},
        **kwargs,
    ) -> Any:
        """
        Make an API request to the Github API.

        This method is used for making API requests to the Github API. It is used internally by the other methods in the client.

        Args:
            - `endpoint (str)`: Name of the endpoint to make the request to.
            - `method (str)`: HTTP method to use for the request.
            - `headers (dict)`: HTTP headers to include in the request.
            - `**kwargs`: Keyword arguments to pass to the endpoint URL.

        Returns:
            - `response (httpx.Response)`: Response from the API request.

        Raises:
            - ImportError: If the `httpx` library is not installed.
            - httpx.HTTPError: If the API request fails.

        Examples:
            >>> response = client.request("getTree", "GET", owner="owner", repo="repo", tree_sha="tree_sha")
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "Please install httpx to use the GithubReader. "
                "You can do so by running `pip install httpx`."
            )

        _headers = {**self.headers, **headers}

        client: httpx.AsyncClient
        async with httpx.AsyncClient(
            headers=_headers, base_url=self.base_url
        ) as client:
            try:
                response = await client.request(
                    method, url=self.__endpoints[endpoint].format(**kwargs)
                )
            except httpx.HTTPError as excp:
                print(f"HTTP Exception for {excp.request.url} - {excp}")
                raise excp
            return response

    async def get_branch(
        self, owner: str, repo: str, branch: str
    ) -> GitBranchResponseModel:
        """
        Get information about a branch. (Github API endpoint: getBranch)

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `branch (str)`: Name of the branch.

        Returns:
            - `branch_info (GitBranchResponseModel)`: Information about the branch.

        Examples:
            >>> branch_info = client.get_branch("owner", "repo", "branch")
        """
        return GitBranchResponseModel.from_json(
            (
                await self.request(
                    "getBranch", "GET", owner=owner, repo=repo, branch=branch
                )
            ).text
        )

    async def get_tree(
        self, owner: str, repo: str, tree_sha: str
    ) -> GitTreeResponseModel:
        """
        Get information about a tree. (Github API endpoint: getTree)

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `tree_sha (str)`: SHA of the tree.

        Returns:
            - `tree_info (GitTreeResponseModel)`: Information about the tree.

        Examples:
            >>> tree_info = client.get_tree("owner", "repo", "tree_sha")
        """
        return GitTreeResponseModel.from_json(
            (
                await self.request(
                    "getTree", "GET", owner=owner, repo=repo, tree_sha=tree_sha
                )
            ).text
        )
