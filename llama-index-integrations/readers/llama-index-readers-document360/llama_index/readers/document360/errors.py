from tenacity import RetryError
from requests.exceptions import HTTPError


class RateLimitException(Exception):
    pass


__all__ = ["RetryError", "HTTPError", "RateLimitException"]
