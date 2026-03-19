import base64
import hashlib
import os
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)

SPCS_TOKEN_PATH = "/snowflake/session/token"


def get_default_spcs_token() -> str:
    """
    Returns the default OAuth session token in SPCS environments.
    """
    with open(SPCS_TOKEN_PATH) as fp:
        return fp.read()


def is_spcs_environment() -> bool:
    """
    Checks if we're running in a Snowpark Container Services environment.
    """
    return (
        os.path.exists(SPCS_TOKEN_PATH) and os.environ.get("SNOWFLAKE_HOST") is not None
    )


def get_spcs_base_url() -> str:
    """
    Returns the base URL for API calls from within SPCS.
    """
    if not is_spcs_environment():
        raise ValueError("Cannot call get_spcs_base_url unless in an SPCS environment.")
    return os.getenv("SNOWFLAKE_HOST")


def generate_sf_jwt(sf_account: str, sf_user: str, sf_private_key_filepath: str) -> str:
    """
    Generate a JWT for Snowflake key-pair authentication.

    Args:
        sf_account: Fully qualified account name (ORG_ID-ACCOUNT_ID).
        sf_user: Snowflake username.
        sf_private_key_filepath: Path to the user's private key PEM file.

    Returns:
        A signed JWT string.

    """
    with open(sf_private_key_filepath, "rb") as pem_in:
        pemlines = pem_in.read()
        private_key = load_pem_private_key(pemlines, None, default_backend())

    public_key_raw = private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )

    sha256hash = hashlib.sha256()
    sha256hash.update(public_key_raw)
    public_key_fp = "SHA256:" + base64.b64encode(sha256hash.digest()).decode("utf-8")

    account = sf_account.upper()
    user = sf_user.upper()
    qualified_username = account + "." + user

    now = datetime.now(timezone.utc)
    lifetime = timedelta(minutes=59)

    payload = {
        "iss": qualified_username + "." + public_key_fp,
        "sub": qualified_username,
        "iat": now,
        "exp": now + lifetime,
    }

    return jwt.encode(payload, key=private_key, algorithm="RS256")
