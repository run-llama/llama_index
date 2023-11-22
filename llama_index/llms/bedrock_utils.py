import logging
from typing import Any, Callable, Sequence

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)

HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"

# Values taken from https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html#model-parameters-claude
COMPLETION_MODELS = {
    "amazon.titan-tg1-large": 8000,
    "amazon.titan-text-express-v1": 8000,
    "ai21.j2-grande-instruct": 8000,
    "ai21.j2-jumbo-instruct": 8000,
    "ai21.j2-mid": 8000,
    "ai21.j2-mid-v1": 8000,
    "ai21.j2-ultra": 8000,
    "ai21.j2-ultra-v1": 8000,
    "cohere.command-text-v14": 4096,
}

# Anthropic models require prompt to start with "Human:" and
# end with "Assistant:"
CHAT_ONLY_MODELS = {
    "anthropic.claude-instant-v1": 100000,
    "anthropic.claude-v1": 100000,
    "anthropic.claude-v2": 100000,
    "meta.llama2-13b-chat-v1": 2048,
}
BEDROCK_FOUNDATION_LLMS = {**COMPLETION_MODELS, **CHAT_ONLY_MODELS}

# Only the following models support streaming as
# per result of Bedrock.Client.list_foundation_models
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html
STREAMING_MODELS = {
    "amazon.titan-tg1-large",
    "amazon.titan-text-express-v1",
    "anthropic.claude-instant-v1",
    "anthropic.claude-v1",
    "anthropic.claude-v2",
    "meta.llama2-13b-chat-v1",
}

# Each bedrock model specifies parameters with a slightly different name
# most of these are passed optionally by the user in kwargs but max tokens
# is a required argument.
PROVIDER_SPECIFIC_PARAM_NAME = {
    "amazon": {"max_tokens": "maxTokenCount"},
    "ai21": {"max_tokens": "maxTokens"},
    "anthropic": {"max_tokens": "max_tokens_to_sample"},
    "cohere": {"max_tokens": "max_tokens"},
    "meta": {"max_tokens": "max_gen_len"},
}

# The response format for each provider is different
PROVIDER_RESPONSE_LOADER = {
    "amazon": lambda x: x["results"][0]["outputText"],
    "ai21": lambda x: x["completions"][0]["data"]["text"],
    "anthropic": lambda x: x["completion"],
    "cohere": lambda x: x["generations"][0]["text"],
    "meta": lambda x: x["generation"],
}

PROVIDER_STREAM_RESPONSE_LOADER = {
    "amazon": lambda x: x["outputText"],
    "anthropic": lambda x: x["completion"],
    "meta": lambda x: x["generation"],
}
logger = logging.getLogger(__name__)


def bedrock_model_to_param_name(model: str, param_name: str) -> str:
    provider = model.split(".")[0]
    model_params = PROVIDER_SPECIFIC_PARAM_NAME[provider]
    return model_params[param_name]


def _create_retry_decorator(client: Any, max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
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


def completion_with_retry(
    client: Any,
    model: str,
    request_body: str,
    max_retries: int,
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(client=client, max_retries=max_retries)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        if stream:
            return client.invoke_model_with_response_stream(
                modelId=model, body=request_body
            )
        return client.invoke_model(modelId=model, body=request_body)

    return _completion_with_retry(**kwargs)


def _message_to_bedrock_prompt(message: ChatMessage) -> str:
    if message.role == MessageRole.USER:
        prompt = f"{HUMAN_PREFIX} {message.content}"
    elif message.role == MessageRole.ASSISTANT:
        prompt = f"{ASSISTANT_PREFIX} {message.content}"
    elif message.role == MessageRole.SYSTEM:
        prompt = f"{HUMAN_PREFIX} <system>{message.content}</system>"
    elif message.role == MessageRole.FUNCTION:
        raise ValueError(f"Message role {MessageRole.FUNCTION} is not supported.")
    else:
        raise ValueError(f"Unknown message role: {message.role}")

    return prompt


def messages_to_bedrock_prompt(messages: Sequence[ChatMessage]) -> str:
    if len(messages) == 0:
        raise ValueError("Got empty list of messages.")

    # NOTE: make sure the prompt ends with the assistant prefix
    if messages[-1].role != MessageRole.ASSISTANT:
        messages = [
            *list(messages),
            ChatMessage(role=MessageRole.ASSISTANT, content=""),
        ]

    str_list = [_message_to_bedrock_prompt(message) for message in messages]
    return "".join(str_list)


def get_request_body(provider: str, prompt: str, inference_paramters: dict) -> dict:
    if provider == "amazon":
        response_body = {
            "inputText": prompt,
            "textGenerationConfig": {**inference_paramters},
        }
    else:
        response_body = {"prompt": prompt, **inference_paramters}
    return response_body


def get_text_from_response(provider: str, response: dict, stream: bool = False) -> str:
    if stream:
        return PROVIDER_STREAM_RESPONSE_LOADER[provider](response)
    return PROVIDER_RESPONSE_LOADER[provider](response)


def completion_to_chat_decorator(
    func: Callable[..., CompletionResponse]
) -> Callable[..., ChatResponse]:
    """Convert a completion function to a chat function."""

    def wrapper(messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # normalize input
        prompt = messages_to_bedrock_prompt(messages)
        completion_response = func(prompt, **kwargs)
        # normalize output
        return completion_response_to_chat_response(completion_response)

    return wrapper


def stream_completion_to_chat_decorator(
    func: Callable[..., CompletionResponseGen]
) -> Callable[..., ChatResponseGen]:
    """Convert a completion function to a chat function."""

    def wrapper(messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        # normalize input
        prompt = messages_to_bedrock_prompt(messages)
        completion_response = func(prompt, **kwargs)
        # normalize output
        return stream_completion_response_to_chat_response(completion_response)

    return wrapper
