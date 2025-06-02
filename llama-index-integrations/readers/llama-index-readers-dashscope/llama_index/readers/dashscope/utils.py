import hashlib
import logging

from enum import Enum
from httpx._models import Response
from typing import Dict, Any, Type, TypeVar
from llama_index.readers.dashscope.domain.base_domains import DictToObject

T = TypeVar("T", bound=DictToObject)

# Asyncio error messages
nest_asyncio_err = "cannot be called from a running event loop"
nest_asyncio_msg = "The event loop is already running. Add `import nest_asyncio; nest_asyncio.apply()` to your code to fix this issue."


def get_stream_logger(name="dashscope-parser", level=logging.INFO, format_string=None):
    if not format_string:
        format_string = "%(asctime)s %(name)s [%(levelname)s] %(thread)d : %(message)s"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(format_string)
    fh = logging.StreamHandler()
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_file_md5(file_path):
    with open(file_path, "rb") as f:
        md5 = hashlib.md5()
        while chunk := f.read(8192):
            md5.update(chunk)
    return md5.hexdigest()


def generate_request_id():
    """Generate a random request id."""
    import uuid

    return str(uuid.uuid4())


def __is_response_successful(response_data: Dict[str, Any]) -> bool:
    """Check if the response data indicates a successful operation."""
    return ("code" in response_data) and (
        response_data["code"] == "Success" or response_data["code"] == "success"
    )


def __raise_exception(response: Response, process: str) -> None:
    """Log the error and raise a specific exception based on the response."""
    error_message = f"Failed to {process}: {response.text}"
    raise ValueError(error_message)


class RetryException(Exception):
    """
    Custom exception class to indicate a situation where an operation needs to be retried.

    This exception should be raised when an operation fails due to anticipated recoverable reasons,
    suggesting to the caller that a retry logic might be appropriate.
    """

    def __init__(
        self, message="Operation failed, requiring a retry", cause=None
    ) -> None:
        """
        Initialize a RetryException instance.

        :param message: Detailed information about the exception, a string by default set as "Operation failed, requiring a retry"
        :param cause: The original exception object that caused this exception, optional
        """
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        """
        Return a string representation of the exception, including the original exception information if present.

        :return: String representation of the exception details
        """
        if self.cause:
            return f"{super().__str__()} caused by: {self.cause}"
        else:
            return super().__str__()


def __raise_exception_for_retry(response: Response, process: str) -> None:
    """Log the error and raise a specific exception based on the response."""
    error_message = f"Failed to {process}: {response.text}"
    raise RetryException(cause=error_message)


logger = get_stream_logger(name="DashScopeResponseHandler")


def dashscope_response_handler(
    response: Response, process: str, result_class: Type[T], url: str = ""
) -> T:
    """Handle the response from the DashScope API."""
    if response is None:
        raise ValueError(
            f"DashScopeParse {process} [URL:{url}] http response object is none."
        )

    if not isinstance(process, str) or not process:
        raise ValueError(
            "DashScopeParse func [dashscope_response_handler] process parameter is empty."
        )

    if response.status_code != 200:
        logger.error(
            f"DashScopeParse {process} [URL:{url}] response http status code is not 200: [{response.status_code}:{response.text}]"
        )
        if response.status_code == 429:
            __raise_exception_for_retry(response, process)
        __raise_exception(response, process)
    try:
        response_data = response.json()
    except Exception as e:
        logger.error(
            f"DashScopeParse {process} [URL:{url}] response data is not json: {response.text}."
        )
        __raise_exception(response, process)

    if not __is_response_successful(response_data):
        logger.error(
            f"DashScopeParse {process} [URL:{url}] response fail: {response.text}."
        )
        __raise_exception(response, process)

    if "data" not in response_data:
        logger.error(
            f"DashScopeParse {process} [URL:{url}] response data does not contain 'data' key: {response_data}."
        )
        __raise_exception(response, process)
    if "request_id" in response_data and process != "query":
        logger.info(
            f"DashScopeParse {process} [URL:{url}] request_id: {response_data['request_id']}."
        )
    return result_class.from_dict(response_data["data"])


class ResultType(Enum):
    """The result type for the parser."""

    DASHSCOPE_DOCMIND = "DASHSCOPE_DOCMIND"


SUPPORTED_FILE_TYPES = [".pdf", ".doc", ".docx"]
