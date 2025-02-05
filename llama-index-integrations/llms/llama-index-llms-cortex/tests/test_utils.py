from unittest.mock import mock_open, patch

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from llama_index.llms.cortex.utils import generate_sf_jwt


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
