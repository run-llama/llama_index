"""Tests for GitHub App authentication."""

import time
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test if PyJWT is available
try:
    import jwt

    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    jwt = None

try:
    from llama_index.readers.github.github_app_auth import (
        GitHubAppAuth,
        GitHubAppAuthenticationError,
    )
    from llama_index.readers.github.repository.github_client import GithubClient
    from llama_index.readers.github.issues.github_client import GitHubIssuesClient
    from llama_index.readers.github.collaborators.github_client import (
        GitHubCollaboratorsClient,
    )
    from llama_index.readers.github import GithubRepositoryReader

    HAS_GITHUB_APP_AUTH = True
except ImportError:
    HAS_GITHUB_APP_AUTH = False


# Sample RSA private key for testing (this is a test key, not a real private key)
# pragma: allowlist secret
TEST_PRIVATE_KEY = os.getenv("TEST_PRIVATE_KEY", "not-a-private-key")


@pytest.mark.skipif(not HAS_JWT, reason="PyJWT not installed")
@pytest.mark.skipif(
    not HAS_GITHUB_APP_AUTH, reason="GitHub App auth module not available"
)
class TestGitHubAppAuth:
    """Test GitHub App authentication class."""

    def test_init_requires_app_id(self):
        """Test that app_id is required."""
        with pytest.raises(GitHubAppAuthenticationError, match="app_id is required"):
            GitHubAppAuth(
                app_id="", private_key=TEST_PRIVATE_KEY, installation_id="123"
            )

    def test_init_requires_private_key(self):
        """Test that private_key is required."""
        with pytest.raises(
            GitHubAppAuthenticationError, match="private_key is required"
        ):
            GitHubAppAuth(app_id="123", private_key="", installation_id="456")

    def test_init_requires_installation_id(self):
        """Test that installation_id is required."""
        with pytest.raises(
            GitHubAppAuthenticationError, match="installation_id is required"
        ):
            GitHubAppAuth(
                app_id="123", private_key=TEST_PRIVATE_KEY, installation_id=""
            )

    def test_init_success(self):
        """Test successful initialization."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        assert auth.app_id == "123456"
        assert auth.private_key == TEST_PRIVATE_KEY
        assert auth.installation_id == "789012"
        assert auth.base_url == "https://api.github.com"
        assert auth._token_cache is None
        assert auth._token_expires_at == 0

    def test_init_custom_base_url(self):
        """Test initialization with custom base URL."""
        auth = GitHubAppAuth(
            app_id="123",
            private_key=TEST_PRIVATE_KEY,
            installation_id="456",
            base_url="https://github.enterprise.com/api/v3",
        )

        assert auth.base_url == "https://github.enterprise.com/api/v3"

    @pytest.mark.skipif(
        condition=TEST_PRIVATE_KEY == "not-a-private-key",
        reason="An SSH private key is not available",
    )
    def test_generate_jwt(self):
        """Test JWT generation."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        token = auth._generate_jwt()

        # Decode the JWT to verify its contents
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert decoded["iss"] == "123456"
        assert "iat" in decoded
        assert "exp" in decoded

        # Check that expiry is approximately 10 minutes from issue time (allow 60s buffer for iat)
        time_diff = decoded["exp"] - decoded["iat"]
        assert 600 <= time_diff <= 660, (
            f"Expected JWT lifespan around 600-660s, got {time_diff}s"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        condition=TEST_PRIVATE_KEY == "not-a-private-key",
        reason="An SSH private key is not available",
    )
    async def test_get_installation_token_success(self):
        """Test successful installation token retrieval."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"token": "ghs_test_token_123"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            token = await auth.get_installation_token()

            assert token == "ghs_test_token_123"
            assert auth._token_cache == "ghs_test_token_123"
            assert auth._token_expires_at > time.time()

            # Verify the API call was made correctly
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert (
                call_args[0][0]
                == "https://api.github.com/app/installations/789012/access_tokens"
            )

    @pytest.mark.asyncio
    async def test_get_installation_token_uses_cache(self):
        """Test that cached token is returned when valid."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        # Set up a cached token that won't expire soon
        auth._token_cache = "cached_token"
        auth._token_expires_at = time.time() + 1000  # Expires in ~16 minutes

        # Should return cached token without making API call
        token = await auth.get_installation_token()

        assert token == "cached_token"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        condition=TEST_PRIVATE_KEY == "not-a-private-key",
        reason="An SSH private key is not available",
    )
    async def test_get_installation_token_refreshes_expired(self):
        """Test that expired token is refreshed."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        # Set up an expired cached token
        auth._token_cache = "expired_token"
        auth._token_expires_at = time.time() - 100  # Expired 100 seconds ago

        mock_response = MagicMock()
        mock_response.json.return_value = {"token": "ghs_new_token_456"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            token = await auth.get_installation_token()

            assert token == "ghs_new_token_456"
            assert auth._token_cache == "ghs_new_token_456"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        condition=TEST_PRIVATE_KEY == "not-a-private-key",
        reason="An SSH private key is not available",
    )
    async def test_get_installation_token_refreshes_when_near_expiry(self):
        """Test that token is refreshed when near expiry (within buffer)."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        # Set up a token that expires within the buffer period (5 minutes)
        auth._token_cache = "expiring_soon_token"
        auth._token_expires_at = time.time() + 200  # Expires in ~3 minutes

        mock_response = MagicMock()
        mock_response.json.return_value = {"token": "ghs_refreshed_token"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            token = await auth.get_installation_token()

            assert token == "ghs_refreshed_token"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        condition=TEST_PRIVATE_KEY == "not-a-private-key",
        reason="An SSH private key is not available",
    )
    async def test_get_installation_token_force_refresh(self):
        """Test force refresh of token."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        # Set up a valid cached token
        auth._token_cache = "valid_token"
        auth._token_expires_at = time.time() + 1000

        mock_response = MagicMock()
        mock_response.json.return_value = {"token": "ghs_forced_refresh_token"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            token = await auth.get_installation_token(force_refresh=True)

            assert token == "ghs_forced_refresh_token"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        condition=TEST_PRIVATE_KEY == "not-a-private-key",
        reason="An SSH private key is not available",
    )
    async def test_get_installation_token_http_error(self):
        """Test handling of HTTP errors."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None

            # Mock HTTP error
            import httpx

            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"

            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Unauthorized", request=MagicMock(), response=mock_response
                )
            )
            mock_client_class.return_value = mock_client

            with pytest.raises(
                GitHubAppAuthenticationError, match="Failed to get installation token"
            ):
                await auth.get_installation_token()

    def test_is_token_valid(self):
        """Test token validity checking."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        # No token cached
        assert not auth._is_token_valid()

        # Token expires in ~6.7 minutes (within 5-minute buffer, should be invalid)
        auth._token_cache = "token"
        auth._token_expires_at = time.time() + 400
        assert auth._is_token_valid()  # 400 seconds > 300 seconds buffer

        # Token expires in 10 minutes (well outside buffer, should be valid)
        auth._token_expires_at = time.time() + 600
        assert auth._is_token_valid()

        # Token expires in 4 minutes (within buffer, should be invalid)
        auth._token_expires_at = time.time() + 240
        assert not auth._is_token_valid()

        # Expired token
        auth._token_expires_at = time.time() - 100
        assert not auth._is_token_valid()

    def test_invalidate_token(self):
        """Test token invalidation."""
        auth = GitHubAppAuth(
            app_id="123456", private_key=TEST_PRIVATE_KEY, installation_id="789012"
        )

        # Set up cached token
        auth._token_cache = "some_token"
        auth._token_expires_at = time.time() + 1000

        # Invalidate
        auth.invalidate_token()

        assert auth._token_cache is None
        assert auth._token_expires_at == 0


@pytest.mark.skipif(not HAS_GITHUB_APP_AUTH, reason="GitHub App auth not available")
class TestGithubClientWithAppAuth:
    """Test GithubClient with GitHub App authentication."""

    def test_init_with_pat(self):
        """Test initialization with PAT (backward compatibility)."""
        client = GithubClient(github_token="ghp_test_token")

        assert client._github_token == "ghp_test_token"
        assert not client._use_github_app
        assert client._github_app_auth is None

    def test_init_with_github_app(self):
        """Test initialization with GitHub App auth."""
        app_auth = GitHubAppAuth(
            app_id="123", private_key=TEST_PRIVATE_KEY, installation_id="456"
        )

        client = GithubClient(github_app_auth=app_auth)

        assert client._github_app_auth is app_auth
        assert client._use_github_app
        assert client._github_token is None

    def test_init_with_both_raises_error(self):
        """Test that providing both PAT and GitHub App auth raises error."""
        app_auth = GitHubAppAuth(
            app_id="123", private_key=TEST_PRIVATE_KEY, installation_id="456"
        )

        with pytest.raises(ValueError, match="Cannot provide both"):
            GithubClient(github_token="ghp_token", github_app_auth=app_auth)

    def test_init_with_neither_raises_error(self):
        """Test that providing neither PAT nor GitHub App auth raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Please provide a Github token"):
                GithubClient()

    @pytest.mark.asyncio
    async def test_get_auth_headers_with_pat(self):
        """Test getting auth headers with PAT."""
        client = GithubClient(github_token="ghp_test_token")

        headers = await client._get_auth_headers()

        assert headers["Authorization"] == "Bearer ghp_test_token"
        assert "Accept" in headers
        assert "X-GitHub-Api-Version" in headers

    @pytest.mark.asyncio
    async def test_get_auth_headers_with_github_app(self):
        """Test getting auth headers with GitHub App."""
        app_auth = GitHubAppAuth(
            app_id="123", private_key=TEST_PRIVATE_KEY, installation_id="456"
        )

        # Mock the get_installation_token method
        app_auth.get_installation_token = AsyncMock(return_value="ghs_app_token_123")

        client = GithubClient(github_app_auth=app_auth)

        headers = await client._get_auth_headers()

        assert headers["Authorization"] == "Bearer ghs_app_token_123"
        assert "Accept" in headers
        assert "X-GitHub-Api-Version" in headers
        app_auth.get_installation_token.assert_called_once()


@pytest.mark.skipif(not HAS_GITHUB_APP_AUTH, reason="GitHub App auth not available")
class TestIssuesClientWithAppAuth:
    """Test GitHubIssuesClient with GitHub App authentication."""

    def test_init_with_github_app(self):
        """Test initialization with GitHub App auth."""
        app_auth = GitHubAppAuth(
            app_id="123", private_key=TEST_PRIVATE_KEY, installation_id="456"
        )

        client = GitHubIssuesClient(github_app_auth=app_auth)

        assert client._github_app_auth is app_auth
        assert client._use_github_app

    def test_init_with_both_raises_error(self):
        """Test that providing both PAT and GitHub App auth raises error."""
        app_auth = GitHubAppAuth(
            app_id="123", private_key=TEST_PRIVATE_KEY, installation_id="456"
        )

        with pytest.raises(ValueError, match="Cannot provide both"):
            GitHubIssuesClient(github_token="ghp_token", github_app_auth=app_auth)


@pytest.mark.skipif(not HAS_GITHUB_APP_AUTH, reason="GitHub App auth not available")
class TestCollaboratorsClientWithAppAuth:
    """Test GitHubCollaboratorsClient with GitHub App authentication."""

    def test_init_with_github_app(self):
        """Test initialization with GitHub App auth."""
        app_auth = GitHubAppAuth(
            app_id="123", private_key=TEST_PRIVATE_KEY, installation_id="456"
        )

        client = GitHubCollaboratorsClient(github_app_auth=app_auth)

        assert client._github_app_auth is app_auth
        assert client._use_github_app

    def test_init_with_both_raises_error(self):
        """Test that providing both PAT and GitHub App auth raises error."""
        app_auth = GitHubAppAuth(
            app_id="123", private_key=TEST_PRIVATE_KEY, installation_id="456"
        )

        with pytest.raises(ValueError, match="Cannot provide both"):
            GitHubCollaboratorsClient(
                github_token="ghp_token", github_app_auth=app_auth
            )


@pytest.mark.skipif(not HAS_GITHUB_APP_AUTH, reason="GitHub App auth not available")
class TestRepositoryReaderWithAppAuth:
    """Test GithubRepositoryReader with GitHub App authentication."""

    def test_reader_with_github_app_client(self):
        """Test creating reader with GitHub App authenticated client."""
        app_auth = GitHubAppAuth(
            app_id="123", private_key=TEST_PRIVATE_KEY, installation_id="456"
        )

        client = GithubClient(github_app_auth=app_auth)
        reader = GithubRepositoryReader(
            github_client=client, owner="test-owner", repo="test-repo"
        )

        assert reader._github_client is client
        assert reader._github_client._use_github_app
