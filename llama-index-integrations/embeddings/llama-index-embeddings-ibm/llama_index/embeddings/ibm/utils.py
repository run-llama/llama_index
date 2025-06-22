import os
import urllib.parse
from typing import Dict, Optional, Union


from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)

# Import SecretStr directly from pydantic
# since there is not one in llama_index.core.bridge.pydantic
try:
    from pydantic.v1 import SecretStr
except ImportError:
    from pydantic import SecretStr


def resolve_watsonx_credentials(
    *,
    url: Optional[str] = None,
    apikey: Optional[str] = None,
    token: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> Dict[str, SecretStr]:
    """
    Resolve watsonx.ai credentials. If the value of given param is None
    then tries to find corresponding environment variable.


    :raises ValueError: raises when value of required attribute is not found
    :return: Dictionary with resolved credentials items
    :rtype: Dict[str, SecretStr]
    """
    creds = {}
    creds["url"] = convert_to_secret_str(
        get_from_param_or_env("url", url, "WATSONX_URL")
    )

    parsed_url = urllib.parse.urlparse(creds["url"].get_secret_value())
    if parsed_url.netloc.endswith("cloud.ibm.com"):
        if not (apikey or "WATSONX_APIKEY" in os.environ) and not (
            token or "WATSONX_TOKEN" in os.environ
        ):
            raise ValueError(
                "Did not find 'apikey' or 'token',"
                " please add an environment variable"
                " `WATSONX_APIKEY` or 'WATSONX_TOKEN' "
                "which contains it,"
                " or pass 'apikey' or 'token'"
                " as a named parameter."
            )
        elif apikey or "WATSONX_APIKEY" in os.environ:
            creds["apikey"] = convert_to_secret_str(
                get_from_param_or_env("apikey", apikey, "WATSONX_APIKEY")
            )
        else:
            creds["token"] = convert_to_secret_str(
                get_from_param_or_env("token", token, "WATSONX_TOKEN")
            )
    else:
        if (
            not token
            and "WATSONX_TOKEN" not in os.environ
            and not password
            and "WATSONX_PASSWORD" not in os.environ
            and not apikey
            and "WATSONX_APIKEY" not in os.environ
        ):
            raise ValueError(
                "Did not find 'token', 'password' or 'apikey',"
                " please add an environment variable"
                " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_APIKEY' "
                "which contains it,"
                " or pass 'token', 'password' or 'apikey'"
                " as a named parameter."
            )
        elif token or "WATSONX_TOKEN" in os.environ:
            creds["token"] = convert_to_secret_str(
                get_from_param_or_env("token", token, "WATSONX_TOKEN")
            )

        elif password or "WATSONX_PASSWORD" in os.environ:
            creds["password"] = convert_to_secret_str(
                get_from_param_or_env("password", password, "WATSONX_PASSWORD")
            )

            creds["username"] = convert_to_secret_str(
                get_from_param_or_env("username", username, "WATSONX_USERNAME")
            )

        elif apikey or "WATSONX_APIKEY" in os.environ:
            creds["apikey"] = convert_to_secret_str(
                get_from_param_or_env("apikey", apikey, "WATSONX_APIKEY")
            )

            creds["username"] = convert_to_secret_str(
                get_from_param_or_env("username", username, "WATSONX_USERNAME")
            )

        if not instance_id or "WATSONX_INSTANCE_ID" not in os.environ:
            creds["instance_id"] = convert_to_secret_str(
                get_from_param_or_env("instance_id", instance_id, "WATSONX_INSTANCE_ID")
            )

    return creds


def convert_to_secret_str(value: Union[SecretStr, str]) -> SecretStr:
    """Convert a string to a SecretStr."""
    if isinstance(value, SecretStr):
        return value
    return SecretStr(value)
