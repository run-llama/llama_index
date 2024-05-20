import os
from typing import Dict, Any, Optional


from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)


def resolve_watsonx_credentials(
    *,
    url: Optional[str] = None,
    apikey: Optional[str] = None,
    token: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    instance_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Resolve watsonx.ai credentials. If the value of given param is None
    then tries to find corresponding environment variable.


    :raises ValueError: raises when value of required attribute is not found
    :return: Dictionary with resolved credentials items
    :rtype: Dict[str, Any]
    """
    creds = {}
    creds["url"] = get_from_param_or_env("url", url, "WATSONX_URL")

    if "cloud.ibm.com" in creds["url"]:
        creds["apikey"] = get_from_param_or_env("apikey", apikey, "WATSONX_APIKEY")
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
            creds["token"] = get_from_param_or_env("token", token, "WATSONX_TOKEN")

        elif password or "WATSONX_PASSWORD" in os.environ:
            creds["password"] = get_from_param_or_env(
                "password", password, "WATSONX_PASSWORD"
            )

            creds["username"] = get_from_param_or_env(
                "username", username, "WATSONX_USERNAME"
            )

        elif apikey or "WATSONX_APIKEY" in os.environ:
            creds["apikey"] = get_from_param_or_env("apikey", apikey, "WATSONX_APIKEY")

            creds["username"] = get_from_param_or_env(
                "username", username, "WATSONX_USERNAME"
            )

        if not instance_id or "WATSONX_INSTANCE_ID" not in os.environ:
            creds["instance_id"] = get_from_param_or_env(
                "instance_id", instance_id, "WATSONX_INSTANCE_ID"
            )

    return creds
