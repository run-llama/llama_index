from typing import Any

from packaging import version


def _import_pinecone() -> Any:
    """
    Try to import pinecone module. If it's not already installed, instruct user how to install.
    """
    try:
        import pinecone
    except ImportError as e:
        raise ImportError(
            "Could not import pinecone python package. "
            "Please install it with `pip install pinecone-client`."
        ) from e
    return pinecone


def _is_pinecone_v3() -> bool:
    """
    Check whether the pinecone client is >= 3.0.0.
    """
    pinecone = _import_pinecone()
    pinecone_client_version = pinecone.__version__
    if version.parse(pinecone_client_version) >= version.parse(
        "3.0.0"
    ):  # Will not work with .dev versions, e.g. "3.0.0.dev8"
        return True
    return False
