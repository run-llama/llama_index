from unittest.mock import mock_open, patch

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from llama_index.llms.cortex.utils import (
    generate_sf_jwt,
    is_spcs_environment,
    get_spcs_base_url,
    get_default_spcs_token,
    SPCS_TOKEN_PATH,
)
import os


def test_spcs_utils():
    os.environ["SNOWFLAKE_HOST"] = "abc-xyz.snowflakecomputing.com"
    # Mock the path check to ensure we're not in SPCS environment
    with patch("os.path.exists", return_value=False):
        # Test that ValueError is raised when not in SPCS environment
        try:
            get_spcs_base_url()
            assert AssertionError("ValueError not raised when not in SPCS environment")
        except ValueError:
            pass

    # Test is_spcs_environment
    with patch("os.path.exists", return_value=True):
        assert is_spcs_environment()
    with patch("os.path.exists", return_value=False):
        assert not is_spcs_environment()

    # Test get_default_spcs_token
    fake_token = "fake-jwt-token-for-testing"
    with patch("builtins.open", mock_open(read_data=fake_token)) as mock_file:
        token = get_default_spcs_token()
        assert token == fake_token
        mock_file.assert_called_once_with(SPCS_TOKEN_PATH)

    del os.environ["SNOWFLAKE_HOST"]


def test_generate_sf_jwt():
    sf_account = "MY_SNOWFLAKE_ORG-MY_SNOWFLAKE_ACCOUNT"
    sf_user = "MY_SNOWFLAKE_USER"

    private_key_obj = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Serialize the private key to PEM format
    private_key = private_key_obj.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    with patch("builtins.open", mock_open(read_data=private_key)) as mock_file:
        mock_file.return_value.read.return_value = (
            private_key  # Ensure binary data is returned
        )
        token = generate_sf_jwt(sf_account, sf_user, "dummy_key_file.pem")

    assert isinstance(token, str)
