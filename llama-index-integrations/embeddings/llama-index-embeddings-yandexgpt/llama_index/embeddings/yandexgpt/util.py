import json
from typing import Optional, Dict


class YException(Exception):
    """Custom exception for YAuth related errors."""


class YAuth:
    """Class to handle authentication using folder ID and API key."""

    def __init__(self, folder_id: str = None, api_key: str = None) -> None:
        """
        Initialize the YAuth object with folder_id and api_key.

        :param folder_id: The folder ID for authentication.
        :param api_key: The API key for authentication.
        """
        self.folder_id = folder_id
        self.api_key = api_key

    @property
    def headers(self) -> Optional[dict]:
        """
        Generate authentication headers.

        :return: A dictionary containing the authorization headers.
        """
        if self.folder_id is not None and self.api_key is not None:
            return {
                "Authorization": f"Api-key {self.api_key}",
                "x-folder-id": self.folder_id,
            }

        return None

    @staticmethod
    def from_dict(js: Dict[str, str]) -> Optional["YAuth"]:
        """
        Create a YAuth instance from a dictionary.

        :param js: A dictionary containing 'folder_id' and 'api_key'.
        :return: A YAuth instance.
        :raises YException: If 'folder_id' or 'api_key' is not provided.
        """
        if js.get("folder_id") is not None and js.get("api_key") is not None:
            return YAuth(js["folder_id"], api_key=js["api_key"])
        raise YException(
            "Cannot create valid authentication object: you need to provide folder_id and either iam token or api_key fields"
        )

    @staticmethod
    def from_config_file(fn: str) -> "YAuth":
        """
        Create a YAuth instance from a configuration file.

        :param fn: Path to the JSON configuration file.
        :return: A YAuth instance.
        """
        with open(fn, encoding="utf-8") as f:
            js = json.load(f)
        return YAuth.from_dict(js)

    @staticmethod
    def from_params(kwargs) -> "YAuth":
        """
        Create a YAuth instance from parameters.

        :param kwargs: A dictionary containing either a 'config' file path or direct 'folder_id' and 'api_key'.
        :return: A YAuth instance.
        """
        if kwargs.get("config") is not None:
            return YAuth.from_config_file(kwargs["config"])
        return YAuth.from_dict(kwargs)
