"""HTTP client for the Famulor Platform v1 REST API."""

import os
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

import requests

DEFAULT_BASE_URL = "https://app.famulor.io/api/v1"
API_KEY_ENV = "FAMULOR_API_KEY"
_SUCCESS_STATUS = frozenset({200, 201})


class FamulorAPIError(Exception):
    """
    Error returned by the Famulor Platform v1 API.

    ``status_code`` is ``0`` when the request did not receive an HTTP response.
    ``code`` and ``message`` come from the documented error envelope when the
    body is JSON. ``retry_after`` is the ``Retry-After`` header, in seconds,
    when the API sends one.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        retry_after: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retry_after = retry_after
        text = f"Famulor API error {status_code} ({code}): {message}"
        if retry_after:
            text = f"{text} Retry after {retry_after} seconds."
        super().__init__(text)


def resolve_api_key(api_key: Optional[str]) -> str:
    """Return the explicit API key or the ``FAMULOR_API_KEY`` environment variable."""
    key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
    if key is None or not str(key).strip():
        raise ValueError(
            "A Famulor API key is required. Pass api_key or set FAMULOR_API_KEY. "
            "Create a key under Settings -> API & MCP at https://app.famulor.io."
        )
    resolved = str(key).strip()
    if any(character.isspace() for character in resolved):
        raise ValueError("api_key must not contain whitespace.")
    return resolved


def normalize_base_url(base_url: str) -> str:
    """
    Validate a Platform v1 base URL.

    The hosted API is ``https://app.famulor.io/api/v1``. A white-label host may
    replace the domain. The path stays ``/api/v1``.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be an https URL ending in /api/v1.")
    cleaned = base_url.strip().rstrip("/")
    parsed = urlsplit(cleaned)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not hostname:
        raise ValueError("base_url must be an https URL ending in /api/v1.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "base_url must not include credentials, a query string, or a fragment."
        )
    if hostname == "famulor.de" or hostname.endswith(".famulor.de"):
        raise ValueError(
            "Famulor 1.0 hosts such as app.famulor.de are not supported. "
            "Use https://app.famulor.io/api/v1 or a white-label host."
        )
    if parsed.path != "/api/v1":
        raise ValueError(
            "base_url path must be /api/v1. Legacy /api/user endpoints are not used."
        )
    return f"https://{parsed.netloc.lower()}{parsed.path}"


class FamulorClient:
    """Minimal JSON client for ``https://app.famulor.io/api/v1``."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a positive number of seconds.")
        if timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds.")
        self.api_key = api_key
        self.base_url = normalize_base_url(base_url)
        self.timeout = float(timeout)

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send one authenticated request and return the JSON object.

        The tenant is taken from the API key. No workspace header is sent.
        Redirects are not followed, so the bearer token stays on this host.
        """
        url = self.base_url + "/" + path.lstrip("/")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        request_kwargs: Dict[str, Any] = {
            "headers": headers,
            "timeout": self.timeout,
            "allow_redirects": False,
        }
        if params:
            request_kwargs["params"] = dict(params)
        if json_body is not None:
            request_kwargs["json"] = dict(json_body)

        try:
            response = requests.request(method, url, **request_kwargs)
        except requests.RequestException as exc:
            raise FamulorAPIError(
                status_code=0,
                code="request_failed",
                message=str(exc),
            ) from exc

        return _parse_response(response)


def _parse_response(response: requests.Response) -> Dict[str, Any]:
    payload: Any
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if response.status_code not in _SUCCESS_STATUS:
        code = "http_error"
        message = response.text or response.reason or "Request failed."
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                if error.get("code"):
                    code = str(error["code"])
                if error.get("message"):
                    message = str(error["message"])
        raise FamulorAPIError(
            status_code=response.status_code,
            code=code,
            message=message,
            retry_after=response.headers.get("Retry-After"),
        )

    if not isinstance(payload, dict):
        raise FamulorAPIError(
            status_code=response.status_code,
            code="invalid_response",
            message="Famulor returned a success status without a JSON object.",
        )
    return payload
