import base64
import hashlib
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)


def generate_sf_jwt(sf_account: str, sf_user: str, sf_private_key_filepath: str) -> str:
    """
    Generate a JSON Web Token for a Snowflake user.

    Args:
        sf_account: Fully qualified snowflake account name (ORG_ID-ACCOUNT_ID).
        sf_user: User to generate token for.
        sf_private_key_filepath: Path to user's private key.

    Returns:
        str: JSON Web Token
    """
    with open(sf_private_key_filepath, "rb") as pem_in:
        pemlines = pem_in.read()
        # TODO: Add support for encrypted private keys
        private_key = load_pem_private_key(pemlines, None, default_backend())

    # Get the raw bytes of the public key.
    public_key_raw = private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )

    # Get the sha256 hash of the raw bytes.
    sha256hash = hashlib.sha256()
    sha256hash.update(public_key_raw)

    # Base64-encode the value and prepend the prefix 'SHA256:'.
    public_key_fp = "SHA256:" + base64.b64encode(sha256hash.digest()).decode("utf-8")

    # Use uppercase for the account identifier and user name.
    account = sf_account.upper()
    user = sf_user.upper()
    qualified_username = account + "." + user

    # Get the current time in order to specify the time when the JWT was issued and the expiration time of the JWT.
    now = datetime.now(timezone.utc)

    # Specify the length of time during which the JWT will be valid. You can specify at most 1 hour.
    lifetime = timedelta(minutes=59)

    # Create the payload for the token.
    payload = {
        # Set the issuer to the fully qualified username concatenated with the public key fingerprint (calculated in the  previous step).
        "iss": qualified_username + "." + public_key_fp,
        # Set the subject to the fully qualified username.
        "sub": qualified_username,
        # Set the issue time to now.
        "iat": now,
        # Set the expiration time, based on the lifetime specified for this object.
        "exp": now + lifetime,
    }

    # Generate the JWT. private_key is the private key that you read from the private key file in the previous step when you generated the public key fingerprint.
    encoding_algorithm = "RS256"
    return jwt.encode(payload, key=private_key, algorithm=encoding_algorithm)
