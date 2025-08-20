"""
GitHub API client for pull requests.
"""

import os
from typing import Any, Dict, Optional, Protocol


class BaseGitHubPullRequestsClient(Protocol):
    def get_all_endpoints(self) -> Dict[str, str]: ...

    async def request(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, Any] = {},
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Any: ...

    async def get_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        sort: str = "created",
        direction: str = "desc",
        page: int = 1,
        per_page: int = 30,
    ) -> Dict: ...

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> Dict: ...

    async def get_pull_request_reviews(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        page: int = 1,
    ) -> Dict: ...

    async def get_pull_request_comments(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        page: int = 1,
    ) -> Dict: ...


class GitHubPullRequestsClient:
    """
    An asynchronous client for interacting with the GitHub API for pull requests.

    The client requires a GitHub token for authentication, which can be passed as an argument
    or set as an environment variable.
    If no GitHub token is provided, the client will raise a ValueError.

    Examples:
        >>> client = GitHubPullRequestsClient("my_github_token")
        >>> prs = client.get_pull_requests("owner", "repo")

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
        Initialize the GitHubPullRequestsClient.

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
                    + "You can do so by passing it as an argument to the GitHubPullRequestsClient,"
                    + "or by setting the GITHUB_TOKEN environment variable."
                )

        self._base_url = base_url
        self._api_version = api_version
        self._verbose = verbose

        self._endpoints = {
            "getPullRequests": "/repos/{owner}/{repo}/pulls",
            "getPullRequest": "/repos/{owner}/{repo}/pulls/{pull_number}",
            "getPullRequestReviews": "/repos/{owner}/{repo}/pulls/{pull_number}/reviews",
            "getPullRequestComments": "/repos/{owner}/{repo}/pulls/{pull_number}/comments",
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
            >>> response = client.request("getPullRequests", "GET",
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

    async def get_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        sort: str = "created",
        direction: str = "desc",
        page: int = 1,
        per_page: int = 30,
    ) -> Dict:
        """
        List pull requests in a repository.

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `state (str)`: State of the pull requests to return.
                Can be: "open", "closed", "all". Default: "open".
            - `sort (str)`: What to sort results by.
                Can be: "created", "updated", "popularity", "long-running". Default: "created".
            - `direction (str)`: Direction to sort.
                Can be: "asc", "desc". Default: "desc".
            - `page (int)`: Page number of the results to fetch (defaults to 1).
            - `per_page (int)`: Number of results per page (max 100, defaults to 30).

        Returns:
            - List of pull request objects.
              See: https://docs.github.com/en/rest/pulls/pulls#list-pull-requests

        Examples:
            >>> prs = client.get_pull_requests("owner", "repo", state="all")

        """
        return (
            await self.request(
                endpoint="getPullRequests",
                method="GET",
                params={
                    "state": state,
                    "sort": sort,
                    "direction": direction,
                    "page": page,
                    "per_page": min(per_page, 100),  # GitHub API max is 100
                },
                owner=owner,
                repo=repo,
            )
        ).json()

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> Dict:
        """
        Get a specific pull request.

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `pull_number (int)`: Number of the pull request.

        Returns:
            - Pull request object with detailed information.
              See: https://docs.github.com/en/rest/pulls/pulls#get-a-pull-request

        Examples:
            >>> pr = client.get_pull_request("owner", "repo", 123)

        """
        return (
            await self.request(
                endpoint="getPullRequest",
                method="GET",
                owner=owner,
                repo=repo,
                pull_number=pull_number,
            )
        ).json()

    async def get_pull_request_reviews(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        page: int = 1,
        per_page: int = 30,
    ) -> Dict:
        """
        List reviews for a pull request.

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `pull_number (int)`: Number of the pull request.
            - `page (int)`: Page number of the results to fetch (defaults to 1).
            - `per_page (int)`: Number of results per page (max 100, defaults to 30).

        Returns:
            - List of review objects.
              See: https://docs.github.com/en/rest/pulls/reviews#list-reviews-for-a-pull-request

        Examples:
            >>> reviews = client.get_pull_request_reviews("owner", "repo", 123)

        """
        return (
            await self.request(
                endpoint="getPullRequestReviews",
                method="GET",
                params={
                    "page": page,
                    "per_page": min(per_page, 100),
                },
                owner=owner,
                repo=repo,
                pull_number=pull_number,
            )
        ).json()

    async def get_pull_request_comments(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        page: int = 1,
        per_page: int = 30,
    ) -> Dict:
        """
        List review comments for a pull request.

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.
            - `pull_number (int)`: Number of the pull request.
            - `page (int)`: Page number of the results to fetch (defaults to 1).
            - `per_page (int)`: Number of results per page (max 100, defaults to 30).

        Returns:
            - List of review comment objects.
              See: https://docs.github.com/en/rest/pulls/comments#list-review-comments-in-a-repository

        Examples:
            >>> comments = client.get_pull_request_comments("owner", "repo", 123)

        """
        return (
            await self.request(
                endpoint="getPullRequestComments",
                method="GET",
                params={
                    "page": page,
                    "per_page": min(per_page, 100),
                },
                owner=owner,
                repo=repo,
                pull_number=pull_number,
            )
        ).json()


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Test the GitHubPullRequestsClient."""
        client = GitHubPullRequestsClient()
        prs = await client.get_pull_requests(
            owner="octocat", repo="Hello-World", state="all"
        )

        for pr in prs[:3]:  # Show first 3 PRs
            print(f"PR #{pr['number']}: {pr['title']}")
            print(f"State: {pr['state']}")
            print(f"Author: {pr['user']['login']}")
            print(f"Created: {pr['created_at']}")
            if pr['merged_at']:
                print(f"Merged: {pr['merged_at']}")
            print("-" * 40)

    asyncio.run(main())
