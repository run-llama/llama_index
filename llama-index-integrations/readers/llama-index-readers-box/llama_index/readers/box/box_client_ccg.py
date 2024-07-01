"""
Handles the box client object creation
orchestrates the authentication process.
"""

import os
import dotenv

from box_sdk_gen import BoxClient
from box_sdk_gen import BoxCCGAuth, CCGConfig
from box_sdk_gen import FileWithInMemoryCacheTokenStorage


ENV_CCG = ".ccg.env"


class BoxConfigCCG:
    """Box client configurations."""

    def __init__(self) -> None:
        dotenv.load_dotenv(ENV_CCG)
        # Common configurations
        self.client_id = os.getenv("BOX_CLIENT_ID")
        self.client_secret = os.getenv("BOX_CLIENT_SECRET")

        # CCG configurations
        self.enterprise_id = os.getenv("BOX_ENTERPRISE_ID")
        self.ccg_user_id = os.getenv("BOX_CCG_USER_ID")

        self.cache_file = os.getenv("BOX_CACHE_FILE", ".ccg.tk")


def __repr__(self) -> str:
    return f"ConfigCCG({self.__dict__})"


def get_ccg_enterprise_client(config: BoxConfigCCG) -> BoxClient:
    """Returns a box sdk Client object."""
    ccg = CCGConfig(
        client_id=config.client_id,
        client_secret=config.client_secret,
        enterprise_id=config.enterprise_id,
        token_storage=FileWithInMemoryCacheTokenStorage(".ent" + config.cache_file),
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
        token_storage=FileWithInMemoryCacheTokenStorage(".user" + config.cache_file),
    )
    auth = BoxCCGAuth(ccg)
    auth.with_user_subject(user_id)

    return BoxClient(auth)

    # return client
