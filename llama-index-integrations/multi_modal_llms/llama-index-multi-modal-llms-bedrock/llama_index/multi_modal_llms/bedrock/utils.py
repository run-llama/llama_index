import base64
import logging
from typing import Any, Dict, Optional, Sequence
import filetype
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.multi_modal_llms.generic_utils import encode_image
from llama_index.core.schema import ImageDocument

DEFAULT_BEDROCK_REGION = "us-east-1"

# Only include multi-modal capable models
BEDROCK_MULTI_MODAL_MODELS = {
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "anthropic.claude-3-opus-20240229-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
}

MISSING_CREDENTIALS_ERROR_MESSAGE = """No AWS credentials found.
Please set up your AWS credentials using one of the following methods:
1. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables
2. Configure AWS CLI credentials
3. Use IAM role-based authentication
"""

logger = logging.getLogger(__name__)


def infer_image_mimetype_from_base64(base64_string) -> str:
    decoded_data = base64.b64decode(base64_string)
    kind = filetype.guess(decoded_data)
    return kind.mime if kind is not None else None


def infer_image_mimetype_from_file_path(image_file_path: str) -> str:
    file_extension = image_file_path.split(".")[-1].lower()

    if file_extension in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif file_extension == "png":
        return "image/png"
    elif file_extension == "gif":
        return "image/gif"
    elif file_extension == "webp":
        return "image/webp"

    return "image/jpeg"


def generate_bedrock_multi_modal_message(
    prompt: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
) -> Dict[str, Any]:
    """Generate message for Bedrock multi-modal API."""
    if image_documents is None:
        return {"role": "user", "content": [{"type": "text", "text": prompt}]}

    message_content = []
    # Add text content first
    message_content.append({"type": "text", "text": prompt})

    # Add image content
    for image_document in image_documents:
        image_content = {}
        if image_document.image_path:
            base64_image = encode_image(image_document.image_path)
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",  # Default to JPEG
                    "data": base64_image,
                },
            }
        elif "file_path" in image_document.metadata:
            base64_image = encode_image(image_document.metadata["file_path"])
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",  # Default to JPEG
                    "data": base64_image,
                },
            }
        elif image_document.image:
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",  # Default to JPEG
                    "data": image_document.image,
                },
            }

        if image_content:
            message_content.append(image_content)

    return {"role": "user", "content": message_content}


def resolve_bedrock_credentials(
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], str]:
    """Resolve AWS Bedrock credentials.

    The order of precedence is:
    1. Explicitly passed credentials
    2. Environment variables
    3. Default region
    """
    region = get_from_param_or_env(
        "region_name", region_name, "AWS_REGION", DEFAULT_BEDROCK_REGION
    )
    access_key = get_from_param_or_env(
        "aws_access_key_id", aws_access_key_id, "AWS_ACCESS_KEY_ID", ""
    )
    secret_key = get_from_param_or_env(
        "aws_secret_access_key", aws_secret_access_key, "AWS_SECRET_ACCESS_KEY", ""
    )

    return access_key, secret_key, region


def _create_retry_decorator(client: Any, max_retries: int) -> Any:
    """Create a retry decorator for Bedrock API calls."""
    min_seconds = 4
    max_seconds = 10
    try:
        import boto3  # noqa
    except ImportError as e:
        raise ImportError(
            "You must install the `boto3` package to use Bedrock."
            "Please `pip install boto3`"
        ) from e

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(client.exceptions.ThrottlingException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _create_retry_decorator_async(max_retries: int) -> Any:
    """Create a retry decorator for async Bedrock API calls."""
    min_seconds = 4
    max_seconds = 10
    try:
        import aioboto3  # noqa
    except ImportError as e:
        raise ImportError(
            "You must install the `aioboto3` package to use async Bedrock."
            "Please `pip install aioboto3`"
        ) from e

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type()),  # TODO: Add throttling exception for async
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def invoke_model_with_retry(
    client: Any,
    model: str,
    messages: Dict[str, Any],
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the model invocation."""
    retry_decorator = _create_retry_decorator(client=client, max_retries=max_retries)

    @retry_decorator
    def _invoke_with_retry(**kwargs: Any) -> Any:
        return client.invoke_model(**kwargs)

    return _invoke_with_retry(
        modelId=model,
        body=messages,
        **kwargs,
    )


async def invoke_model_with_retry_async(
    session: Any,
    config: Any,
    model: str,
    messages: Dict[str, Any],
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the model invocation asynchronously."""
    retry_decorator = _create_retry_decorator_async(max_retries=max_retries)

    @retry_decorator
    async def _invoke_with_retry(**kwargs: Any) -> Any:
        async with session.client("bedrock-runtime", config=config) as client:
            return await client.invoke_model(**kwargs)

    return await _invoke_with_retry(
        modelId=model,
        body=messages,
        **kwargs,
    )
