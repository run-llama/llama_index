import logging
from functools import wraps
from typing import Any, Callable

from packaging import version

MIN_ADS_VERSION = "2.12.9"

logger = logging.getLogger(__name__)


class UnsupportedOracleAdsVersionError(Exception):
    """
    Custom exception for unsupported `oracle-ads` versions.

    Attributes:
        current_version (str): The installed version of `oracle-ads`.
        required_version (str): The minimum required version of `oracle-ads`.

    """

    def __init__(self, current_version: str, required_version: str):
        """
        Initialize the UnsupportedOracleAdsVersionError.

        Args:
            current_version (str): The currently installed version of `oracle-ads`.
            required_version (str): The minimum required version of `oracle-ads`.

        """
        super().__init__(
            f"The `oracle-ads` version {current_version} currently installed is incompatible with "
            "the `llama-index-llms-oci-data-science` version in use. To resolve this issue, "
            f"please upgrade to `oracle-ads:{required_version}` or later using the "
            "command: `pip install oracle-ads -U`"
        )


def _validate_dependency(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to validate the presence and version of the `oracle-ads` package.

    This decorator checks whether `oracle-ads` is installed and ensures its version meets
    the minimum requirement. If not, it raises an appropriate error.

    Args:
        func (Callable[..., Any]): The function to wrap with the dependency validation.

    Returns:
        Callable[..., Any]: The wrapped function.

    Raises:
        ImportError: If `oracle-ads` is not installed.
        UnsupportedOracleAdsVersionError: If the installed version is below the required version.

    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            from ads import __version__ as ads_version

            if version.parse(ads_version) < version.parse(MIN_ADS_VERSION):
                raise UnsupportedOracleAdsVersionError(ads_version, MIN_ADS_VERSION)
        except ImportError as ex:
            raise ImportError(
                "Could not import `oracle-ads` Python package. "
                "Please install it with `pip install oracle-ads`."
            ) from ex

        return func(*args, **kwargs)

    return wrapper
