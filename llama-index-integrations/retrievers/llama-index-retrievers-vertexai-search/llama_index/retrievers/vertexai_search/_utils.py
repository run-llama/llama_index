"""Utilities to init Vertex AI."""

from importlib import metadata
from typing import Optional

from google.api_core.gapic_v1.client_info import ClientInfo


def get_user_agent(module: Optional[str] = None) -> str:
    r"""
    Returns a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.

    Returns:
        Tuple[str, str]

    """
    try:
        llama_index_version = metadata.version("llama-index")
    except metadata.PackageNotFoundError:
        llama_index_version = "0.0.0"
    client_library_version = (
        f"{llama_index_version}-{module}" if module else llama_index_version
    )
    return (client_library_version, f"llama-index/{client_library_version}")


def get_client_info(module: Optional[str] = None) -> "ClientInfo":
    r"""
    Returns a client info object with a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.

    Returns:
        google.api_core.gapic_v1.client_info.ClientInfo

    """
    client_library_version, user_agent = get_user_agent(module)
    return ClientInfo(
        client_library_version=client_library_version,
        user_agent=user_agent,
    )
