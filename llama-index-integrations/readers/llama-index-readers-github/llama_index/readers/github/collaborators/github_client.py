"""
GitHub API client for collaborators.
"""

import os
from typing import Any, Dict, Optional, Protocol


class BaseGitHubCollaboratorsClient(Protocol):
    def get_all_endpoints(self) -> Dict[str, str]:
        ...

    async def request(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, Any] = {},
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Any:
        ...

    async def get_collaborators(
        self,
        owner: str,
        repo: str,
        page: int = 1,
    ) -> Dict:
        ...


class GitHubCollaboratorsClient:
    """
    An asynchronous client for interacting with the GitHub API for collaborators.

    The client requires a GitHub token for authentication, which can be passed as an argument
    or set as an environment variable.
    If no GitHub token is provided, the client will raise a ValueError.

    Examples:
        >>> client = GitHubCollaboratorsClient("my_github_token")
        >>> collaborators = client.get_collaborators("owner", "repo")
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
        Initialize the GitHubCollaboratorsClient.

        Args:
            - github_token (str): GitHub token for authentication.
                If not provided, the client will try to get it from
                the GITHUB_TOKEN environment variable.
            - base_url (str): Base URL for the GitHub API
                (defaults to "https://api.github.com").
            - api_version (str): GitHub API version (defaults to "2022-11-28").

        Raises:
            ValueError: If no GitHub token is provided.
        """
        if github_token is None:
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token is None:
                raise ValueError(
                    "Please provide a GitHub token. "
                    + "You can do so by passing it as an argument to the GitHubReader,"
                    + "or by setting the GITHUB_TOKEN environment variable."
                )

        self._base_url = base_url
        self._api_version = api_version
        self._verbose = verbose

        self._endpoints = {
            "getCollaborators": "/repos/{owner}/{repo}/collaborators",
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
            - `**kwargs`: Keyword arguments to pass to the endpoint URL.

        Returns:
            - `response (httpx.Response)`: Response from the API request.

        Raises:
            - ImportError: If the `httpx` library is not installed.
            - httpx.HTTPError: If the API request fails.

        Examples:
            >>> response = client.request("getCollaborators", "GET",
                                owner="owner", repo="repo", state="all")
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "`https` package not found, please run `pip install httpx`"
            )

        _headers = {**self._headers, **headers}

        _client: httpx.AsyncClient
        async with httpx.AsyncClient(
            headers=_headers, base_url=self._base_url, params=params
        ) as _client:
            try:
                response = await _client.request(
                    method, url=self._endpoints[endpoint].format(**kwargs)
                )
                response.raise_for_status()
            except httpx.HTTPError as excp:
                print(f"HTTP Exception for {excp.request.url} - {excp}")
                raise excp  # noqa: TRY201
            return response

    async def get_collaborators(
        self,
        owner: str,
        repo: str,
        page: int = 1,
    ) -> Dict:
        """
        List collaborators in a repository.

        Args:
            - `owner (str)`: Owner of the repository.
            - `repo (str)`: Name of the repository.

        Returns:
            - See https://docs.github.com/en/rest/collaborators/collaborators?apiVersion=2022-11-28#list-repository-collaborators

        Examples:
            >>> repo_collaborators = client.get_collaborators("owner", "repo")
        """
        return (
            await self.request(
                endpoint="getCollaborators",
                method="GET",
                params={
                    "per_page": 100,
                    "page": page,
                },
                owner=owner,
                repo=repo,
            )
        ).json()


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Test the GitHubCollaboratorsClient."""
        client = GitHubCollaboratorsClient()
        collaborators = await client.get_collaborators(
            owner="moncho",
            repo="dry",
        )

        for collab in collaborators:
            print(collab[0]["login"])
            print(collab[0]["email"])

    asyncio.run(main())
