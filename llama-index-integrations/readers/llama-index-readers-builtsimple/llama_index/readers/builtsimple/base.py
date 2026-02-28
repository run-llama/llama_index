"""Base utilities for Built-Simple research API readers."""

from typing import Any, Dict, Optional
from types import TracebackType
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class BuiltSimpleAPIError(Exception):
    """Exception raised when Built-Simple API returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class BuiltSimpleBaseClient:
    """
    Base client for Built-Simple research APIs.

    Provides common functionality for API communication including:
    - Session management with connection pooling
    - Automatic retries with exponential backoff
    - Optional API key authentication
    - Consistent error handling
    """

    DEFAULT_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 0.5

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the base client.

        Args:
            base_url: Base URL for the API (without trailing slash)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds (default: 30)

        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session: Optional[requests.Session] = None

    @property
    def session(self) -> requests.Session:
        """Get or create a requests session with retry logic."""
        if self._session is None:
            self._session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=self.MAX_RETRIES,
                backoff_factor=self.BACKOFF_FACTOR,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            # Set default headers
            self._session.headers.update(
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "llama-index-readers-builtsimple/0.1.0",
                }
            )

            # Add API key if provided
            if self.api_key:
                self._session.headers["Authorization"] = f"Bearer {self.api_key}"

        return self._session

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and extract JSON data.

        Args:
            response: The requests Response object

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            BuiltSimpleAPIError: If the API returns an error

        """
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_msg = f"API request failed: {e}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = f"API error: {error_data['error']}"
                elif "message" in error_data:
                    error_msg = f"API error: {error_data['message']}"
            except (ValueError, KeyError):
                pass
            raise BuiltSimpleAPIError(error_msg, response.status_code) from e

        try:
            return response.json()
        except ValueError as e:
            raise BuiltSimpleAPIError(
                f"Invalid JSON response: {response.text[:200]}"
            ) from e

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a GET request to the API.

        Args:
            endpoint: API endpoint (will be appended to base_url)
            params: Optional query parameters

        Returns:
            Parsed JSON response

        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug(f"GET {url} params={params}")

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise BuiltSimpleAPIError(f"Request failed: {e}") from e

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request to the API.

        Args:
            endpoint: API endpoint (will be appended to base_url)
            data: JSON body data
            params: Optional query parameters

        Returns:
            Parsed JSON response

        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug(f"POST {url} data={data}")

        try:
            response = self.session.post(
                url, json=data, params=params, timeout=self.timeout
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise BuiltSimpleAPIError(f"Request failed: {e}") from e

    def close(self) -> None:
        """Close the session and release resources."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self) -> "BuiltSimpleBaseClient":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()


def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize text content.

    Args:
        text: Raw text that may contain extra whitespace or be None

    Returns:
        Cleaned text string (empty string if input is None)

    """
    if text is None:
        return ""
    # Normalize whitespace while preserving paragraph structure
    lines = text.strip().split("\n")
    cleaned_lines = [" ".join(line.split()) for line in lines]
    return "\n".join(cleaned_lines)


def format_authors(authors: Any) -> str:
    """
    Format authors list into a readable string.

    Args:
        authors: Authors data (can be list, string, or None)

    Returns:
        Formatted author string

    """
    if authors is None:
        return ""
    if isinstance(authors, str):
        return authors
    if isinstance(authors, list):
        # Handle list of author objects or strings
        author_names = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name") or author.get("full_name") or str(author)
                author_names.append(name)
            else:
                author_names.append(str(author))
        return ", ".join(author_names)
    return str(authors)
