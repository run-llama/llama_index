"""Unit tests for shared utilities related to credential parsing (get_username_password)."""

import base64
import hashlib
import hmac
import json
from contextlib import nullcontext

import pytest
from azure.core.credentials import AccessToken, TokenCredential

from llama_index.vector_stores.azure_postgres.common import BasicAuth
from llama_index.vector_stores.azure_postgres.common._shared import (
    TOKEN_CREDENTIAL_SCOPE,
    get_username_password,
)


class TestGetUsernamePassword:
    """Test suite for get_username_password covering BasicAuth, TokenCredential, invalid inputs, and JWT-like token payload extraction."""

    def test_it_works(self, credentials: BasicAuth | TokenCredential) -> None:
        """Ensure username/password extraction works for both credential types."""
        if isinstance(credentials, BasicAuth):
            username, password = get_username_password(credentials)
            assert username == credentials.username, (
                "Username should match BasicAuth username"
            )
            assert password == credentials.password, (
                "Password should match BasicAuth password"
            )
        elif isinstance(credentials, TokenCredential):
            token = credentials.get_token(TOKEN_CREDENTIAL_SCOPE)
            username, password = get_username_password(token)
            assert len(username) > 0, "Username should not be empty for TokenCredential"
            assert password == token.token, (
                "Password should match TokenCredential token"
            )

    def test_invalid_credentials_type(self) -> None:
        """Assert passing an invalid type raises a TypeError."""
        with pytest.raises(TypeError, match="Invalid credentials type"):
            get_username_password("invalid_credentials_type")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ["payload", "username"],
        [
            ({"upn": "test_user_1"}, nullcontext("test_user_1")),
            ({"unique_name": "test_user_2"}, nullcontext("test_user_2")),
            (
                {"upn": "test_user_3", "unique_name": "test_user_4"},
                nullcontext("test_user_3"),
            ),
            (
                {"no-upn-or-unique_name": "test_user_5"},
                pytest.raises(
                    ValueError, match="User name not found in JWT token header"
                ),
            ),
        ],
        ids=[
            "only-upn",
            "only-unique_name",
            "upn-over-unique_name",
            "no-upn-or-unique_name",
        ],
    )
    def test_mock_it_works(
        self, payload: dict, username: nullcontext | pytest.RaisesExc
    ) -> None:
        """Validate extraction from JWT-like access token payloads."""
        _header = {"alg": "HS256", "typ": "JWT"}
        _header_encoded = base64.urlsafe_b64encode(
            json.dumps(_header).encode()
        ).decode()
        _payload_encoded = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode()
        h = hmac.new(
            b"secret",
            ".".join([_header_encoded, _payload_encoded]).encode(),
            hashlib.sha256,
        )
        _signature = base64.urlsafe_b64encode(h.digest()).decode()
        token = AccessToken(
            ".".join([_header_encoded, _payload_encoded, _signature]), -1
        )
        with username as expected_username:
            username_, password = get_username_password(token)
            assert username_ == expected_username, (
                "Username should match expected username from JWT token"
            )
            assert password == token.token, (
                "Password should match TokenCredential token"
            )
