"""
GitHub API client for commits.
"""

import os
from typing import Any, Dict, Optional, Protocol


class BaseGitHubCommitsClient(Protocol):
    def get_all_endpoints(self) -> Dict[str, str]: ...

    async def request(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, Any] = {},
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Any: ...

    async def get_commits(
        self,
        owner: str,
        repo: str,
        sha: Optional[str] = None,
        path: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        page: int = 1,
    ) -> Dict: ...

    async def get_commit(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> Dict: ...


class GitHubCommitsClient:
    """
    An asynchronous client for interacting with the GitHub API for commits.

    The client requires a GitHub token for authentication, which can be passed as an argument
    or set as an environment variable.
    If no GitHub token is provided, the client will raise a ValueError.

    Examples:
        >>> client = GitHubCommitsClient("my_github_token")
        >>> commits = client.get_commits("owner", "repo")

    """

    DEFAULT_BASE_URL = "https://api.github.com"
    DEFAULT_API_VERSION = "2022-11-28"

    def __init__(
        self,
        github_token: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the GitHubCommitsClient.

        Args:
            - github_token (str): GitHub token for authentication.
                If not provided, the client will try to get it from
                the GITHUB_TOKEN environment variable.
            - base_url (str): Base URL for the GitHub API
                (defaults to "https://api.github.com").
            - api_version (str): GitHub API version (defaults to "2022-11-28").
            - verbose (bool): Whether to print verbose output.

        Raises:
            ValueError: If no GitHub token is provided.

        """
        if github_token is None:
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token is None:
                raise ValueError(
                    "Please provide a GitHub token. "
                    + "You can do so by passing it as an argument to the GitHubCommitsClient,"
                    + "or by setting the GITHUB_TOKEN environment variable."
                )

        self._base_url = base_url
        self._api_version = api_version
        self._verbose = verbose

        self._endpoints = {
            "getCommits": "/repos/{owner}/{repo}/commits",
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
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Any:
        """
        Makes an API request to the GitHub API.

        Args:
            - `endpoint (str)`: Name of the endpoint to make the request to.
            - `method (str)`: HTTP method to use for the request.
            - `headers (dict)`: HTTP headers to include in the request.
            - `params (dict)`: Query parameters to include in the request.
            - `**kwargs`: Keyword arguments to pass to the endpoint URL.

        Returns:
            - `response (httpx.Response)`: Response from the API request.

        Raises:
            - ImportError: If the `httpx` library is not installed.
            - httpx.HTTPError: If the API request fails.

        Examples:
            >>> response = client.request("getCommits", "GET",
                                owner="owner", repo="repo")

        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "`httpx` package not found, please run `pip install httpx`"
            )

        _headers = {**self._headers, **headers}

        _client: httpx.AsyncClient
        async with httpx.AsyncClient(
            headers=_headers,
            base_url=self._base_url,
            params=params,
            follow_redirects=True,
        ) as _client:
            try:
                response = await _client.request(
                    method, url=self._endpoints[endpoint].format(**kwargs)
                )
                response.raise_for_status()
            except httpx.HTTPError as excp:
                if self._verbose:
                    print(f"HTTP Exception for {excp.request.url} - {excp}")
                raise excp
            return response

    async def get_commits(
        self,
        owner: str,
        repo: str,
        sha: Optional[str] = None,
        path: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        page: int = 1,
        per_page: int = 30,
    ) -> Dict:
        """
        List commits in a repository.

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `sha (str, optional)`: SHA or branch to start listing commits from.
            - `path (str, optional)`: Only commits containing this file path will be returned.
            - `since (str, optional)`: Only show notifications updated after the given time.
                This is a timestamp in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ.
            - `until (str, optional)`: Only commits before this date will be returned.
                This is a timestamp in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ.
            - `page (int)`: Page number of the results to fetch (defaults to 1).
            - `per_page (int)`: Number of results per page (max 100, defaults to 30).

        Returns:
            - List of commit objects. See: https://docs.github.com/en/rest/commits/commits#list-commits

        Examples:
            >>> commits = client.get_commits("owner", "repo")
            >>> recent_commits = client.get_commits("owner", "repo", since="2024-01-01T00:00:00Z")

        """
        params = {
            "page": page,
            "per_page": min(per_page, 100),  # GitHub API max is 100
        }

        if sha:
            params["sha"] = sha
        if path:
            params["path"] = path
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        return (
            await self.request(
                endpoint="getCommits",
                method="GET",
                params=params,
                owner=owner,
                repo=repo,
            )
        ).json()

    async def get_commit(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> Dict:
        """
        Get a specific commit by SHA.

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `commit_sha (str)`: SHA of the commit to retrieve.

        Returns:
            - Commit object with detailed information including files changed.
              See: https://docs.github.com/en/rest/commits/commits#get-a-commit

        Examples:
            >>> commit = client.get_commit("owner", "repo", "commit_sha")

        """
        return (
            await self.request(
                endpoint="getCommit",
                method="GET",
                owner=owner,
                repo=repo,
                commit_sha=commit_sha,
            )
        ).json()


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Test the GitHubCommitsClient."""
        client = GitHubCommitsClient()
        commits = await client.get_commits(owner="octocat", repo="Hello-World")

        for commit in commits[:3]:  # Show first 3 commits
            print(f"SHA: {commit['sha']}")
            print(f"Message: {commit['commit']['message']}")
            print(f"Author: {commit['commit']['author']['name']}")
            print(f"Date: {commit['commit']['author']['date']}")
            print("-" * 40)

    asyncio.run(main())
