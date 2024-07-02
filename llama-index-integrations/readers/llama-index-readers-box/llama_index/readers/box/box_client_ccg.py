"""
Handles the box client object creation
orchestrates the authentication process.
"""

import os
import dotenv

from box_sdk_gen import BoxClient
from box_sdk_gen import BoxCCGAuth, CCGConfig
from box_sdk_gen import FileWithInMemoryCacheTokenStorage


class BoxConfigCCG:
    """Box client configurations."""

    def __init__(self) -> None:
        dotenv.load_dotenv()
        # Common configurations
        self.client_id = os.getenv("BOX_CLIENT_ID")
        self.client_secret = os.getenv("BOX_CLIENT_SECRET")

        # CCG configurations
        self.enterprise_id = os.getenv("BOX_ENTERPRISE_ID")
        self.ccg_user_id = os.getenv("BOX_CCG_USER_ID")


def get_ccg_enterprise_client(config: BoxConfigCCG) -> BoxClient:
    """Returns a box sdk Client object."""
    ccg = CCGConfig(
        client_id=config.client_id,
        client_secret=config.client_secret,
        enterprise_id=config.enterprise_id,
        token_storage=FileWithInMemoryCacheTokenStorage(".enterprise"),
    )
    auth = BoxCCGAuth(ccg)

    return BoxClient(auth)

    # return client


def get_ccg_user_client(config: BoxConfigCCG, user_id: str) -> BoxClient:
    """Returns a box sdk Client object."""
    ccg = CCGConfig(
        client_id=config.client_id,
        client_secret=config.client_secret,
        user_id=user_id,
        token_storage=FileWithInMemoryCacheTokenStorage(".user"),
    )
    auth = BoxCCGAuth(ccg)
    auth.with_user_subject(user_id)

    return BoxClient(auth)


def reader_box_client_ccg(
    box_client_id: str,
    box_client_secret: str,
    box_enterprise_id: str = None,
    box_user_id: str = None,
) -> BoxClient:
    """
    Creates a BoxClient instance using CCG authentication.

    Args:
        box_client_id: Client ID for Box API access.
        box_client_secret: Client secret for Box API access.
        box_enterprise_id: Optional enterprise ID for enterprise authentication.
        box_user_id: Optional user ID for user authentication.

    Returns:
        A BoxClient instance authenticated with CCG.
    """
    token_storage_filename = ".enterprise" if box_user_id is None else ".user"
    ccg = CCGConfig(
        client_id=box_client_id,
        client_secret=box_client_secret,
        enterprise_id=box_enterprise_id,
        user_id=box_user_id,
        token_storage=FileWithInMemoryCacheTokenStorage(
            filename=token_storage_filename
        ),
    )

    auth = BoxCCGAuth(ccg)
    if box_user_id:
        auth.with_user_subject(box_user_id)
    return BoxClient(auth)
