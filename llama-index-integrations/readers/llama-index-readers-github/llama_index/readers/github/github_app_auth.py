"""
GitHub App authentication module.

This module provides GitHub App authentication support for the GitHub readers.
It handles JWT generation and installation access token management with automatic
token refresh and caching.
"""

import time
from typing import Optional

try:
    import jwt
except ImportError:
    jwt = None  # type: ignore


class GitHubAppAuthenticationError(Exception):
    """Raised when GitHub App authentication fails."""


class GitHubAppAuth:
    """
    GitHub App authentication handler.

    This class manages authentication for GitHub Apps by generating JWTs and
    obtaining/caching installation access tokens. Tokens are automatically
    refreshed when they expire.

    Attributes:
        app_id (str): The GitHub App ID.
        private_key (str): The private key for the GitHub App (PEM format).
        installation_id (str): The installation ID for the GitHub App.

    Examples:
        >>> # Read private key from file
        >>> with open("private-key.pem", "r") as f:
        ...     private_key = f.read()
        >>>
        >>> # Create auth handler
        >>> auth = GitHubAppAuth(
        ...     app_id="123456",
        ...     private_key=private_key,
        ...     installation_id="789012"
        ... )
        >>>
        >>> # Get installation token (cached and auto-refreshed)
        >>> import asyncio
        >>> token = asyncio.run(auth.get_installation_token())

    """

    # Token expiry buffer in seconds (refresh 5 minutes before expiry)
    TOKEN_EXPIRY_BUFFER = 300
    # JWT expiry time in seconds (10 minutes, max allowed by GitHub)
    JWT_EXPIRY_SECONDS = 600
    # Installation token expiry time in seconds (1 hour, GitHub default)
    INSTALLATION_TOKEN_EXPIRY_SECONDS = 3600

    def __init__(
        self,
        app_id: str,
        private_key: str,
        installation_id: str,
        base_url: str = "https://api.github.com",
    ) -> None:
        """
        Initialize GitHubAppAuth.

        Args:
            app_id: The GitHub App ID.
            private_key: The private key for the GitHub App in PEM format.
            installation_id: The installation ID for the GitHub App.
            base_url: Base URL for GitHub API (default: "https://api.github.com").

        Raises:
            ImportError: If PyJWT is not installed.
            GitHubAppAuthenticationError: If initialization fails.

        """
        if jwt is None:
            raise ImportError(
                "PyJWT is required for GitHub App authentication. "
                "Install it with: pip install 'PyJWT[crypto]>=2.8.0'"
            )

        if not app_id:
            raise GitHubAppAuthenticationError("app_id is required")
        if not private_key:
            raise GitHubAppAuthenticationError("private_key is required")
        if not installation_id:
            raise GitHubAppAuthenticationError("installation_id is required")

        self.app_id = app_id
        self.private_key = private_key
        self.installation_id = installation_id
        self.base_url = base_url.rstrip("/")

        # Token cache
        self._token_cache: Optional[str] = None
        self._token_expires_at: float = 0

    def _generate_jwt(self) -> str:
        """
        Generate JWT for GitHub App authentication.

        The JWT is used to authenticate as the GitHub App itself, before
        obtaining an installation access token.

        Returns:
            The generated JWT token.

        Raises:
            GitHubAppAuthenticationError: If JWT generation fails.

        """
        try:
            now = int(time.time())
            payload = {
                "iat": now - 60,  # Issued at (with 60s buffer for clock skew)
                "exp": now + self.JWT_EXPIRY_SECONDS,  # Expires in 10 minutes
                "iss": self.app_id,  # Issuer is the app ID
            }

            return jwt.encode(payload, self.private_key, algorithm="RS256")
        except Exception as e:
            raise GitHubAppAuthenticationError(f"Failed to generate JWT: {e!s}") from e

    async def get_installation_token(self, force_refresh: bool = False) -> str:
        """
        Get or refresh installation access token.

        This method returns a cached token if it's still valid, or requests
        a new token from GitHub if the cached token is expired or about to expire.

        Args:
            force_refresh: If True, forces a token refresh even if cached token
                         is still valid.

        Returns:
            A valid installation access token.

        Raises:
            GitHubAppAuthenticationError: If token retrieval fails.
            ImportError: If httpx is not installed.

        """
        # Check if cached token is still valid (with buffer)
        if not force_refresh and self._is_token_valid():
            return self._token_cache  # type: ignore

        # Generate new token
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for GitHub App authentication. "
                "Install it with: pip install httpx>=0.26.0"
            )

        jwt_token = self._generate_jwt()

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {jwt_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        url = f"{self.base_url}/app/installations/{self.installation_id}/access_tokens"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                self._token_cache = data["token"]
                # Token typically expires in 1 hour
                self._token_expires_at = (
                    time.time() + self.INSTALLATION_TOKEN_EXPIRY_SECONDS
                )

                return self._token_cache
        except httpx.HTTPStatusError as e:
            raise GitHubAppAuthenticationError(
                f"Failed to get installation token: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            raise GitHubAppAuthenticationError(
                f"Failed to get installation token: {e!s}"
            ) from e

    def _is_token_valid(self) -> bool:
        """
        Check if the cached token is still valid.

        Returns:
            True if token exists and is not expired (accounting for buffer).

        """
        if not self._token_cache:
            return False

        # Check if token will expire within the buffer period
        time_until_expiry = self._token_expires_at - time.time()
        return time_until_expiry > self.TOKEN_EXPIRY_BUFFER

    def invalidate_token(self) -> None:
        """
        Invalidate the cached token.

        This forces the next call to get_installation_token() to fetch a new token.
        Useful if you know the token has been revoked or is no longer valid.

        """
        self._token_cache = None
        self._token_expires_at = 0
