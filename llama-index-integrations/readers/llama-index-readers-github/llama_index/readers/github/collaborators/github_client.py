"""
GitHub API client for collaborators.
"""

import os
from typing import Any, Dict, Optional, Protocol, Union

try:
    from llama_index.readers.github.github_app_auth import GitHubAppAuth
except ImportError:
    GitHubAppAuth = None  # type: ignore


class BaseGitHubCollaboratorsClient(Protocol):
    def get_all_endpoints(self) -> Dict[str, str]: ...

    async def request(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, Any] = {},
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Any: ...

    async def get_collaborators(
        self,
        owner: str,
        repo: str,
        page: int = 1,
    ) -> Dict: ...


class GitHubCollaboratorsClient:
    """
    An asynchronous client for interacting with the GitHub API for collaborators.

    The client supports two authentication methods:
    1. Personal Access Token (PAT) - passed as github_token or via GITHUB_TOKEN env var
    2. GitHub App - passed as github_app_auth parameter

    Examples:
        >>> # Using Personal Access Token
        >>> client = GitHubCollaboratorsClient("my_github_token")
        >>> collaborators = client.get_collaborators("owner", "repo")
        >>>
        >>> # Using GitHub App
        >>> from llama_index.readers.github.github_app_auth import GitHubAppAuth
        >>> app_auth = GitHubAppAuth(app_id="123", private_key=key, installation_id="456")
        >>> client = GitHubCollaboratorsClient(github_app_auth=app_auth)

    """

    DEFAULT_BASE_URL = "https://api.github.com"
    DEFAULT_API_VERSION = "2022-11-28"

    def __init__(
        self,
        github_token: Optional[str] = None,
        github_app_auth: Optional[Union["GitHubAppAuth", Any]] = None,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the GitHubCollaboratorsClient.

        Args:
            - github_token (str, optional): GitHub token for authentication.
                If not provided, the client will try to get it from
                the GITHUB_TOKEN environment variable. Mutually exclusive with github_app_auth.
            - github_app_auth (GitHubAppAuth, optional): GitHub App authentication handler.
                Mutually exclusive with github_token.
            - base_url (str): Base URL for the GitHub API
                (defaults to "https://api.github.com").
            - api_version (str): GitHub API version (defaults to "2022-11-28").
            - verbose (bool): Whether to print verbose output (defaults to False).

        Raises:
            ValueError: If neither github_token nor github_app_auth is provided,
                       or if both are provided.

        """
        # Validate authentication parameters
        if github_token is not None and github_app_auth is not None:
            raise ValueError(
                "Cannot provide both github_token and github_app_auth. "
                "Please use only one authentication method."
            )

        self._base_url = base_url
        self._api_version = api_version
        self._verbose = verbose
        self._github_app_auth = github_app_auth
        self._github_token = None

        # Set up authentication
        if github_app_auth is not None:
            self._use_github_app = True
        else:
            self._use_github_app = False
            if github_token is None:
                github_token = os.getenv("GITHUB_TOKEN")
                if github_token is None:
                    raise ValueError(
                        "Please provide a GitHub token or GitHub App authentication. "
                        + "You can pass github_token as an argument, "
                        + "set the GITHUB_TOKEN environment variable, "
                        + "or pass github_app_auth for GitHub App authentication."
                    )
            self._github_token = github_token

        self._endpoints = {
            "getCollaborators": "/repos/{owner}/{repo}/collaborators",
        }

        # Base headers (Authorization header will be added per-request)
        self._base_headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": f"{self._api_version}",
        }

        # For backward compatibility, keep _headers with PAT token
        if not self._use_github_app:
            self._headers = {
                **self._base_headers,
                "Authorization": f"Bearer {self._github_token}",
            }
        else:
            self._headers = self._base_headers.copy()

    def get_all_endpoints(self) -> Dict[str, str]:
        """Get all available endpoints."""
        return {**self._endpoints}

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if self._use_github_app:
            token = await self._github_app_auth.get_installation_token()
            return {
                **self._base_headers,
                "Authorization": f"Bearer {token}",
            }
        else:
            return self._headers

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

        # Get authentication headers (may fetch fresh token for GitHub App)
        auth_headers = await self._get_auth_headers()
        _headers = {**auth_headers, **headers}

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
