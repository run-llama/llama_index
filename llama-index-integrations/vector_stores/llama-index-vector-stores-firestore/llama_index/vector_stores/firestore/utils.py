from importlib.metadata import version
from typing import Optional

from google.cloud import firestore
from google.cloud.firestore_v1.services.firestore.transports.base import (
    DEFAULT_CLIENT_INFO,
)

try:
    __version__ = version("llama-index-vector-store-firestore")
except Exception:
    __version__ = "unknown"

USER_AGENT = "llama-index-vector-store-firestore-python:vectorstore" + __version__


def client_with_user_agent(
    client: Optional[firestore.Client] = None,
) -> firestore.Client:
    """Create a Firestore client with a user agent."""
    client_info = DEFAULT_CLIENT_INFO
    user_agent = USER_AGENT
    client_info.user_agent = user_agent

    if not client:
        client = firestore.Client(client_info=client_info)
    client_agent = client._client_info.user_agent

    if not client_agent:
        client._client_info.user_agent = user_agent
    elif user_agent not in client_agent:
        client._client_info.user_agent = f"{client_agent} {user_agent}"
    return client
