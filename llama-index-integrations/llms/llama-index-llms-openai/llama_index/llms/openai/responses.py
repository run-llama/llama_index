import functools
import httpx
import tiktoken
import base64
from openai import AsyncOpenAI, AzureOpenAI
from openai import OpenAI as SyncOpenAI
from openai.types.responses import (
    Response,
    ResponseStreamEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFileSearchCallCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseTextDeltaEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseComputerToolCall,
    ResponseReasoningItem,
    ResponseCodeInterpreterToolCall,
    ResponseImageGenCallPartialImageEvent,
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall, McpCall
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
    cast,
)

import llama_index.core.instrumentation as instrument
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
    ContentBlock,
    TextBlock,
    ImageBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.bridge.pydantic import (
    Field,
    PrivateAttr,
)
from llama_index.core.constants import (
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection, Model
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.prompts import PromptTemplate
from llama_index.core.program.utils import FlexibleModel
from llama_index.llms.openai.utils import (
    O1_MODELS,
    create_retry_decorator,
    is_function_calling_model,
    openai_modelname_to_contextsize,
    resolve_openai_credentials,
    resolve_tool_choice,
    to_openai_message_dicts,
)


dispatcher = instrument.get_dispatcher(__name__)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def llm_retry_decorator(f: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(f)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        max_retries = getattr(self, "max_retries", 0)
        if max_retries <= 0:
            return f(self, *args, **kwargs)

        retry = create_retry_decorator(
            max_retries=max_retries,
            random_exponential=True,
            stop_after_delay_seconds=60,
            min_seconds=1,
            max_seconds=20,
        )
        return retry(f)(self, *args, **kwargs)

    return wrapper


@runtime_checkable
class Tokenizer(Protocol):
    """Tokenizers support an encode function that returns a list of ints."""

    def encode(self, text: str) -> List[int]:  # fmt: skip
        ...


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = [
        block for block in response.message.blocks if isinstance(block, ToolCallBlock)
    ]
    if len(tool_calls) > 1:
        response.message.blocks = [
            block
            for block in response.message.blocks
            if not isinstance(block, ToolCallBlock)
        ] + [tool_calls[0]]


class OpenAIResponses(FunctionCallingLLM):
    """
    OpenAI Responses LLM.

    Args:
        model: name of the OpenAI model to use.
        temperature: a float from 0 to 1 controlling randomness in generation; higher will lead to more creative, less deterministic responses.
        max_output_tokens: the maximum number of tokens to generate.
        reasoning_options: Optional dictionary to configure reasoning for O1 models.
                    Corresponds to the 'reasoning' parameter in the OpenAI API.
                    Example: {"effort": "low", "summary": "concise"}
        include: Additional output data to include in the model response.
        instructions: Instructions for the model to follow.
        track_previous_responses: Whether to track previous responses. If true, the LLM class will statefully track previous responses.
        store: Whether to store previous responses in OpenAI's storage.
        built_in_tools: The built-in tools to use for the model to augment responses.
        truncation: Whether to auto-truncate the input if it exceeds the model's context window.
        user: An optional identifier to help track the user's requests for abuse.
        strict: Whether to enforce strict validation of the structured output.
        additional_kwargs: Add additional parameters to OpenAI request body.
        max_retries: How many times to retry the API call if it fails.
        timeout: How long to wait, in seconds, for an API call before failing.
        api_key: Your OpenAI api key
        api_base: The base URL of the API to call
        api_version: the version of the API to call
        default_headers: override the default headers for API requests.
        http_client: pass in your own httpx.Client instance.
        async_http_client: pass in your own httpx.AsyncClient instance.

    Examples:
        `pip install llama-index-llms-openai`