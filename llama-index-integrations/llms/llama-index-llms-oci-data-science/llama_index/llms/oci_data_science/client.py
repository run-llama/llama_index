import asyncio
import functools
import json
import logging
import time
from abc import ABC
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import httpx
import oci
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 3
TIMEOUT = 600  # Timeout in seconds
STATUS_FORCE_LIST = [429, 500, 502, 503, 504]
DEFAULT_ENCODING = "utf-8"

_T = TypeVar("_T", bound="BaseClient")

logger = logging.getLogger(__name__)


class OCIAuth(httpx.Auth):
    """
    Custom HTTPX authentication class that uses the OCI Signer for request signing.

    Attributes:
        signer (oci.signer.Signer): The OCI signer used to sign requests.
    """

    def __init__(self, signer: oci.signer.Signer):
        """
        Initialize the OCIAuth instance.

        Args:
            signer (oci.signer.Signer): The OCI signer to use for signing requests.
        """
        self.signer = signer

    def auth_flow(self, request: httpx.Request) -> Iterator[httpx.Request]:
        """
        The authentication flow that signs the HTTPX request using the OCI signer.

        Args:
            request (httpx.Request): The outgoing HTTPX request to be signed.

        Yields:
            httpx.Request: The signed HTTPX request.
        """
        # Create a requests.Request object from the HTTPX request
        req = requests.Request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            data=request.content,
        )
        prepared_request = req.prepare()

        # Sign the request using the OCI Signer
        self.signer.do_request_sign(prepared_request)

        # Update the original HTTPX request with the signed headers
        request.headers.update(prepared_request.headers)

        # Proceed with the request
        yield request


class ExtendedRequestException(Exception):
    """
    Custom exception for handling request errors with additional context.

    Attributes:
        original_exception (Exception): The original exception that caused the error.
        response_text (str): The text of the response received from the request, if available.
    """

    def __init__(self, message: str, original_exception: Exception, response_text: str):
        """
        Initialize the ExtendedRequestException.

        Args:
            message (str): The error message associated with the exception.
            original_exception (Exception): The original exception that caused the error.
            response_text (str): The text of the response received from the request, if available.
        """
        super().__init__(message)
        self.original_exception = original_exception
        self.response_text = response_text


def _should_retry_exception(e: ExtendedRequestException) -> bool:
    """
    Determine whether the exception should trigger a retry.

    Args:
        e (ExtendedRequestException): The exception raised.

    Returns:
        bool: True if the exception should trigger a retry, False otherwise.
    """
    original_exception = e.original_exception if hasattr(e, "original_exception") else e
    if isinstance(original_exception, httpx.HTTPStatusError):
        return original_exception.response.status_code in STATUS_FORCE_LIST
    elif isinstance(original_exception, httpx.RequestError):
        return True
    return False


def _create_retry_decorator(
    max_retries: int,
    backoff_factor: float,
    random_exponential: bool = False,
    stop_after_delay_seconds: Optional[float] = None,
    min_seconds: float = 0,
    max_seconds: float = 60,
) -> Callable[[Any], Any]:
    """
    Create a tenacity retry decorator with the specified configuration.

    Args:
        max_retries (int): The maximum number of retry attempts.
        backoff_factor (float): The backoff factor for calculating retry delays.
        random_exponential (bool): Whether to use random exponential backoff.
        stop_after_delay_seconds (Optional[float]): Maximum total time to retry.
        min_seconds (float): Minimum wait time between retries.
        max_seconds (float): Maximum wait time between retries.

    Returns:
        Callable[[Any], Any]: A tenacity retry decorator configured with the specified strategy.
    """
    wait_strategy = (
        wait_random_exponential(min=min_seconds, max=max_seconds)
        if random_exponential
        else wait_exponential(
            multiplier=backoff_factor, min=min_seconds, max=max_seconds
        )
    )

    stop_strategy = stop_after_attempt(max_retries)
    if stop_after_delay_seconds is not None:
        stop_strategy = stop_strategy | stop_after_delay(stop_after_delay_seconds)

    retry_strategy = retry_if_exception(_should_retry_exception)
    return retry(
        wait=wait_strategy,
        stop=stop_strategy,
        retry=retry_strategy,
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _retry_decorator(f: Callable) -> Callable:
    """
    Decorator to apply retry logic to a function using tenacity.

    Args:
        f (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function with retry logic applied.
    """

    @functools.wraps(f)
    def wrapper(self, *args: Any, **kwargs: Any):
        retries = getattr(self, "retries", DEFAULT_RETRIES)
        if retries <= 0:
            return f(self, *args, **kwargs)
        backoff_factor = getattr(self, "backoff_factor", DEFAULT_BACKOFF_FACTOR)
        retry_func = _create_retry_decorator(
            max_retries=retries,
            backoff_factor=backoff_factor,
            random_exponential=False,
            stop_after_delay_seconds=getattr(self, "timeout", TIMEOUT),
            min_seconds=0,
            max_seconds=60,
        )

        return retry_func(f)(self, *args, **kwargs)

    return wrapper


class BaseClient(ABC):
    """
    Base class for invoking models via HTTP requests with retry logic.

    Attributes:
        endpoint (str): The URL endpoint to send the request.
        auth (Any): The authentication signer for the requests.
        retries (int): The number of retry attempts for the request.
        backoff_factor (float): The factor to determine the delay between retries.
        timeout (Union[float, Tuple[float, float]]): The timeout setting for the HTTP request.
        kwargs (Dict): Additional keyword arguments.
    """

    def __init__(
        self,
        endpoint: str,
        auth: Optional[Any] = None,
        retries: Optional[int] = DEFAULT_RETRIES,
        backoff_factor: Optional[float] = DEFAULT_BACKOFF_FACTOR,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the BaseClient.

        Args:
            endpoint (str): The URL endpoint to send the request.
            auth (Optional[Any]): The authentication signer for the requests.
            retries (Optional[int]): The number of retry attempts for the request.
            backoff_factor (Optional[float]): The factor to determine the delay between retries.
            timeout (Optional[Union[float, Tuple[float, float]]]): The timeout setting for the HTTP request.
            **kwargs: Additional keyword arguments.
        """
        self.endpoint = endpoint
        self.retries = retries or DEFAULT_RETRIES
        self.backoff_factor = backoff_factor or DEFAULT_BACKOFF_FACTOR
        self.timeout = timeout or TIMEOUT
        self.kwargs = kwargs

        # Use default signer from ADS if `auth` not provided
        if not auth:
            try:
                from ads.common import auth as authutil

                auth = auth or authutil.default_signer()
            except ImportError as ex:
                raise ImportError(
                    "The authentication signer for the requests was not provided. "
                    "Use `auth` attribute to provide the signer. "
                    "The authentication methods supported for LlamaIndex are equivalent to those "
                    "used with other OCI services and follow the standard SDK authentication methods, "
                    "specifically API Key, session token, instance principal, and resource principal. "
                    "For more details, refer to the documentation: "
                    "`https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html`. "
                    "Alternatively you can use the `oracle-ads` package. "
                    "Please install it with `pip install oracle-ads` and follow the example provided here: "
                    "`https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html#authentication`."
                ) from ex

        # Validate auth object
        if not callable(auth.get("signer")):
            raise ValueError("Auth object must have a 'signer' callable attribute.")
        self.auth = OCIAuth(auth["signer"])

        logger.debug(
            f"Initialized {self.__class__.__name__} with endpoint={self.endpoint}, "
            f"retries={self.retries}, backoff_factor={self.backoff_factor}, timeout={self.timeout}"
        )

    def _parse_streaming_line(
        self, line: Union[bytes, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single line from the streaming response.

        Args:
            line (Union[bytes, str]): A line of the response in bytes or string format.

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON object, or None if the line is to be ignored.

        Raises:
            Exception: Raised if the line contains an error object.
            json.JSONDecodeError: Raised if the line cannot be decoded as JSON.
        """
        logger.debug(f"Parsing streaming line: {line}")

        if isinstance(line, bytes):
            line = line.decode(DEFAULT_ENCODING)

        line = line.strip()

        if line.lower().startswith("data:"):
            line = line[5:].lstrip()

        if not line or line.startswith("[DONE]"):
            logger.debug("Received end of stream signal or empty line.")
            return None

        try:
            json_line = json.loads(line)
            logger.debug(f"Parsed JSON line: {json_line}")
        except json.JSONDecodeError as e:
            logger.debug(f"Error decoding JSON from line: {line}")
            raise json.JSONDecodeError(
                f"Error decoding JSON from line: {e!s}", e.doc, e.pos
            ) from e

        if json_line.get("object") == "error":
            # Raise an error for error objects in the stream
            error_message = json_line.get("message", "Unknown error")
            logger.debug(f"Error in streaming response: {error_message}")
            raise Exception(f"Error in streaming response: {error_message}")

        return json_line

    def _prepare_headers(
        self,
        stream: bool,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Construct and return the headers for a request.

        Args:
            stream (bool): Whether to use streaming for the response.
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.

        Returns:
            Dict[str, str]: The prepared headers.
        """
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if stream:
            default_headers["enable-streaming"] = "true"
        if headers:
            default_headers.update(headers)

        logger.debug(f"Prepared headers: {default_headers}")
        return default_headers


class Client(BaseClient):
    """
    Synchronous HTTP client for invoking models with support for request and streaming APIs.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Client.

        Args:
            *args: Positional arguments forwarded to BaseClient.
            **kwargs: Keyword arguments forwarded to BaseClient.
        """
        super().__init__(*args, **kwargs)
        self._client = httpx.Client(timeout=self.timeout)

    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        """Close the underlying HTTPX client."""
        self._client.close()

    def __enter__(self: _T) -> _T:  # noqa: PYI019
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc: Optional[BaseException] = None,
        exc_tb: Optional[TracebackType] = None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @_retry_decorator
    def _request(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send a POST request to the configured endpoint with retry and error handling.

        Args:
            payload (Dict[str, Any]): Parameters for the request payload.
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.

        Returns:
            Dict[str, Any]: Decoded JSON response.

        Raises:
            ExtendedRequestException: Raised when the request fails.
        """
        logger.debug(f"Starting synchronous request with payload: {payload}")
        try:
            response = self._client.post(
                self.endpoint,
                headers=self._prepare_headers(stream=False, headers=headers),
                auth=self.auth,
                json=payload,
            )
            logger.debug(f"Received response with status code: {response.status_code}")
            response.raise_for_status()
            json_response = response.json()
            logger.debug(f"Response JSON: {json_response}")
            return json_response
        except Exception as e:
            last_exception_text = (
                e.response.text if hasattr(e, "response") and e.response else str(e)
            )
            logger.error(
                f"Request failed. Error: {e!s}. Details: {last_exception_text}"
            )
            raise ExtendedRequestException(
                f"Request failed: {e!s}. Details: {last_exception_text}",
                e,
                last_exception_text,
            ) from e

    def _stream(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Iterator[Mapping[str, Any]]:
        """
        Send a POST request expecting a streaming response.

        Args:
            payload (Dict[str, Any]): Parameters for the request payload.
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.

        Yields:
            Mapping[str, Any]: Decoded JSON response line-by-line.

        Raises:
            ExtendedRequestException: Raised when the request fails.
        """
        logger.debug(f"Starting synchronous streaming request with payload: {payload}")
        last_exception_text = None

        for attempt in range(1, self.retries + 2):  # retries + initial attempt
            logger.debug(f"Attempt {attempt} for synchronous streaming request.")
            try:
                with self._client.stream(
                    "POST",
                    self.endpoint,
                    headers=self._prepare_headers(stream=True, headers=headers),
                    auth=self.auth,
                    json={**payload, "stream": True},
                ) as response:
                    try:
                        logger.debug(
                            f"Received streaming response with status code: {response.status_code}"
                        )
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line:  # Skip empty lines
                                continue

                            parsed_line = self._parse_streaming_line(line)
                            if parsed_line:
                                logger.debug(f"Yielding parsed line: {parsed_line}")
                                yield parsed_line
                        return
                    except Exception as e:
                        last_exception_text = (
                            e.response.read().decode(
                                e.response.encoding or DEFAULT_ENCODING
                            )
                            if hasattr(e, "response") and e.response
                            else str(e)
                        )
                        raise

            except Exception as e:
                if attempt <= self.retries and _should_retry_exception(e):
                    delay = self.backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        f"Streaming attempt {attempt} failed: {e}. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Streaming request failed. Error: {e!s}. Details: {last_exception_text}"
                    )
                    raise ExtendedRequestException(
                        f"Streaming request failed: {e!s}. Details: {last_exception_text}",
                        e,
                        last_exception_text,
                    ) from e

    def generate(
        self,
        prompt: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = True,
    ) -> Union[Dict[str, Any], Iterator[Mapping[str, Any]]]:
        """
        Generate text completion for the given prompt.

        Args:
            prompt (str): Input text prompt for the model.
            payload (Optional[Dict[str, Any]]): Additional parameters for the request payload.
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.
            stream (bool): Whether to use streaming for the response.

        Returns:
            Union[Dict[str, Any], Iterator[Mapping[str, Any]]]: A full JSON response or an iterator for streaming responses.
        """
        logger.debug(f"Generating text with prompt: {prompt}, stream: {stream}")
        payload = {**(payload or {}), "prompt": prompt}
        headers = {"route": "/v1/completions", **(headers or {})}
        if stream:
            return self._stream(payload=payload, headers=headers)
        return self._request(payload=payload, headers=headers)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = True,
    ) -> Union[Dict[str, Any], Iterator[Mapping[str, Any]]]:
        """
        Perform a chat interaction with the model.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries for chat interaction.
            payload (Optional[Dict[str, Any]]): Additional parameters for the request payload.
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.
            stream (bool): Whether to use streaming for the response.

        Returns:
            Union[Dict[str, Any], Iterator[Mapping[str, Any]]]: A full JSON response or an iterator for streaming responses.
        """
        logger.debug(f"Starting chat with messages: {messages}, stream: {stream}")
        payload = {**(payload or {}), "messages": messages}
        headers = {"route": "/v1/chat/completions", **(headers or {})}
        if stream:
            return self._stream(payload=payload, headers=headers)
        return self._request(payload=payload, headers=headers)


class AsyncClient(BaseClient):
    """
    Asynchronous HTTP client for invoking models with support for request and streaming APIs, including retry logic.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the AsyncClient.

        Args:
            *args: Positional arguments forwarded to BaseClient.
            **kwargs: Keyword arguments forwarded to BaseClient.
        """
        super().__init__(*args, **kwargs)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    def is_closed(self) -> bool:
        return self._client.is_closed

    async def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        await self._client.aclose()

    async def __aenter__(self: _T) -> _T:  # noqa: PYI019
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc: Optional[BaseException] = None,
        exc_tb: Optional[TracebackType] = None,
    ) -> None:
        await self.close()

    def __del__(self) -> None:
        try:
            if not self._client.is_closed:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
        except Exception:
            pass

    @_retry_decorator
    async def _request(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send a POST request to the configured endpoint with retry and error handling.

        Args:
            payload (Dict[str, Any]): Parameters for the request payload.
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.

        Returns:
            Dict[str, Any]: Decoded JSON response.

        Raises:
            ExtendedRequestException: Raised when the request fails.
        """
        logger.debug(f"Starting asynchronous request with payload: {payload}")
        try:
            response = await self._client.post(
                self.endpoint,
                headers=self._prepare_headers(stream=False, headers=headers),
                auth=self.auth,
                json=payload,
            )
            logger.debug(f"Received response with status code: {response.status_code}")
            response.raise_for_status()
            json_response = response.json()
            logger.debug(f"Response JSON: {json_response}")
            return json_response
        except Exception as e:
            last_exception_text = (
                e.response.text if hasattr(e, "response") and e.response else str(e)
            )
            logger.error(
                f"Request failed. Error: {e!s}. Details: {last_exception_text}"
            )
            raise ExtendedRequestException(
                f"Request failed: {e!s}. Details: {last_exception_text}",
                e,
                last_exception_text,
            ) from e

    async def _stream(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> AsyncIterator[Mapping[str, Any]]:
        """
        Send a POST request expecting a streaming response with retry logic.

        Args:
            payload (Dict[str, Any]): Parameters for the request payload.
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.

        Yields:
            Mapping[str, Any]: Decoded JSON response line-by-line.

        Raises:
            ExtendedRequestException: Raised when the request fails.
        """
        logger.debug(f"Starting asynchronous streaming request with payload: {payload}")
        last_exception_text = None
        for attempt in range(1, self.retries + 2):  # retries + initial attempt
            logger.debug(f"Attempt {attempt} for asynchronous streaming request.")
            try:
                async with self._client.stream(
                    "POST",
                    self.endpoint,
                    headers=self._prepare_headers(stream=True, headers=headers),
                    auth=self.auth,
                    json={**payload, "stream": True},
                ) as response:
                    try:
                        logger.debug(
                            f"Received streaming response with status code: {response.status_code}"
                        )
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:  # Skip empty lines
                                continue
                            parsed_line = self._parse_streaming_line(line)
                            if parsed_line:
                                logger.debug(f"Yielding parsed line: {parsed_line}")
                                yield parsed_line
                        return
                    except Exception as e:
                        if hasattr(e, "response") and e.response:
                            content = await e.response.aread()
                            last_exception_text = content.decode(
                                e.response.encoding or DEFAULT_ENCODING
                            )
                        raise
            except Exception as e:
                if attempt <= self.retries and _should_retry_exception(e):
                    delay = self.backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        f"Streaming attempt {attempt} failed: {e}. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Streaming request failed. Error: {e!s}. Details: {last_exception_text}"
                    )
                    raise ExtendedRequestException(
                        f"Streaming request failed: {e!s}. Details: {last_exception_text}",
                        e,
                        last_exception_text,
                    ) from e

    async def generate(
        self,
        prompt: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[Mapping[str, Any]]]:
        """
        Generate text completion for the given prompt.

        Args:
            prompt (str): Input text prompt for the model.
            payload (Optional[Dict[str, Any]]): Additional parameters for the request payload.
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.
            stream (bool): Whether to use streaming for the response.

        Returns:
            Union[Dict[str, Any], AsyncIterator[Mapping[str, Any]]]: A full JSON response or an async iterator for streaming responses.
        """
        logger.debug(f"Generating text with prompt: {prompt}, stream: {stream}")
        payload = {**(payload or {}), "prompt": prompt}
        headers = {"route": "/v1/completions", **(headers or {})}
        if stream:
            return self._stream(payload=payload, headers=headers)
        return await self._request(payload=payload, headers=headers)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[Mapping[str, Any]]]:
        """
        Perform a chat interaction with the model.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries for chat interaction.
            payload (Optional[Dict[str, Any]]): Additional parameters for the request payload.
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.
            stream (bool): Whether to use streaming for the response.

        Returns:
            Union[Dict[str, Any], AsyncIterator[Mapping[str, Any]]]: A full JSON response or an async iterator for streaming responses.
        """
        logger.debug(f"Starting chat with messages: {messages}, stream: {stream}")
        payload = {**(payload or {}), "messages": messages}
        headers = {"route": "/v1/chat/completions", **(headers or {})}
        if stream:
            return self._stream(payload=payload, headers=headers)
        return await self._request(payload=payload, headers=headers)
