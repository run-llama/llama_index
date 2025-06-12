import logging
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Union,
    Optional,
    Type,
    Tuple,
)
import typing

import google.genai.types as types
import google.genai
import httpx
from google.genai import _transformers
from google.genai import errors

from llama_index.core.bridge.pydantic import BaseModel, ValidationError
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ImageBlock,
    MessageRole,
    TextBlock,
    DocumentBlock,
)
from llama_index.core.program.utils import _repair_incomplete_json
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)
from tenacity.stop import stop_base

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

logger = logging.getLogger(__name__)

ROLES_TO_GEMINI: dict[MessageRole, MessageRole] = {
    MessageRole.USER: MessageRole.USER,
    MessageRole.ASSISTANT: MessageRole.MODEL,
    ## Gemini chat mode only has user and model roles. Put the rest in user role.
    MessageRole.SYSTEM: MessageRole.USER,
    MessageRole.MODEL: MessageRole.MODEL,
    ## Gemini has function role, but chat mode only accepts user and model roles.
    ## https://medium.com/@smallufo/openai-vs-gemini-function-calling-a664f7f2b29f
    ## Agent response's 'tool/function' role is converted to 'user' role.
    MessageRole.TOOL: MessageRole.USER,
    MessageRole.FUNCTION: MessageRole.USER,
}

ROLES_FROM_GEMINI: dict[str, MessageRole] = {
    ## Gemini has user, model and function roles.
    "user": MessageRole.USER,
    "model": MessageRole.ASSISTANT,
    "function": MessageRole.TOOL,
}


def merge_neighboring_same_role_messages(
    messages: Sequence[ChatMessage],
) -> Sequence[ChatMessage]:
    if len(messages) < 2:
        # Nothing to merge
        return messages

    # Gemini does not support multiple messages of the same role in a row, so we merge them
    # However, tool messages are not merged, so that we can keep the tool names
    # (merging them would destroy the tool name)
    merged_messages = []
    i = 0

    while i < len(messages):
        # operate on a copy of the message to avoid mutating the original
        current_message = messages[i].model_copy()
        # Initialize merged content with current message content
        merged_content = current_message.blocks

        # Check if the next message exists and has the same role
        while (
            i + 1 < len(messages)
            and ROLES_TO_GEMINI[messages[i + 1].role]
            == ROLES_TO_GEMINI[current_message.role]
            and current_message.role != MessageRole.TOOL
        ):
            i += 1
            next_message = messages[i]
            merged_content.extend(next_message.blocks)

        # Create a new ChatMessage or similar object with merged content
        merged_message = ChatMessage(
            role=ROLES_TO_GEMINI[current_message.role],
            blocks=merged_content,
            additional_kwargs=current_message.additional_kwargs,
        )

        merged_messages.append(merged_message)
        i += 1

    return merged_messages


def _error_if_finished_early(candidate: types.Candidate) -> None:
    if finish_reason := candidate.finish_reason:
        if finish_reason != types.FinishReason.STOP:
            reason = finish_reason.name

            # Safety reasons have more detail, so include that if we can.
            if finish_reason == types.FinishReason.SAFETY and candidate.safety_ratings:
                relevant_safety = list(
                    filter(
                        lambda sr: sr.probability
                        and sr.probability.value
                        > types.HarmProbability.NEGLIGIBLE.value,
                        candidate.safety_ratings,
                    )
                )
                reason += f" {relevant_safety}"

            raise RuntimeError(f"Response was terminated early: {reason}")


def chat_from_gemini_response(
    response: types.GenerateContentResponse,
) -> ChatResponse:
    if not response.candidates:
        raise ValueError("Response has no candidates")

    top_candidate = response.candidates[0]
    _error_if_finished_early(top_candidate)

    response_feedback = (
        response.prompt_feedback.model_dump() if response.prompt_feedback else {}
    )
    raw = {
        **(top_candidate.model_dump()),
        **response_feedback,
    }
    if response.usage_metadata:
        raw["usage_metadata"] = response.usage_metadata.model_dump()

    content_blocks = []
    if (
        len(response.candidates) > 0
        and response.candidates[0].content
        and response.candidates[0].content.parts
    ):
        parts = response.candidates[0].content.parts
        for part in parts:
            if part.text:
                content_blocks.append(TextBlock(text=part.text))
            if part.inline_data:
                content_blocks.append(
                    ImageBlock(
                        image=part.inline_data.data,
                        image_mimetype=part.inline_data.mime_type,
                    )
                )

    additional_kwargs: Dict[str, Any] = {}
    if response.function_calls:
        for fn in response.function_calls:
            if "tool_calls" not in additional_kwargs:
                additional_kwargs["tool_calls"] = []
            additional_kwargs["tool_calls"].append(fn)

    role = ROLES_FROM_GEMINI[top_candidate.content.role]
    return ChatResponse(
        message=ChatMessage(
            role=role, blocks=content_blocks, additional_kwargs=additional_kwargs
        ),
        raw=raw,
        additional_kwargs=additional_kwargs,
    )


def chat_message_to_gemini(message: ChatMessage) -> types.Content:
    """Convert ChatMessages to Gemini-specific history, including ImageDocuments."""
    parts = []
    for block in message.blocks:
        if isinstance(block, TextBlock):
            if block.text:
                parts.append(types.Part.from_text(text=block.text))
        elif isinstance(block, ImageBlock):
            base64_bytes = block.resolve_image(as_base64=False).read()
            if not block.image_mimetype:
                # TODO: fail ?
                block.image_mimetype = "image/png"

            parts.append(
                types.Part.from_bytes(
                    data=base64_bytes,
                    mime_type=block.image_mimetype,
                )
            )
        elif isinstance(block, DocumentBlock):
            file_buffer = block.resolve_document()
            file_bytes = file_buffer.read()
            mimetype = (
                block.document_mimetype
                if block.document_mimetype is not None
                else "application/pdf"
            )
            parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mimetype))
        else:
            msg = f"Unsupported content block type: {type(block).__name__}"
            raise ValueError(msg)

    for tool_call in message.additional_kwargs.get("tool_calls", []):
        if isinstance(tool_call, dict):
            parts.append(
                types.Part.from_function_call(
                    name=tool_call.get("name"), args=tool_call.get("args")
                )
            )
        else:
            parts.append(
                types.Part.from_function_call(name=tool_call.name, args=tool_call.args)
            )

    # the tool call id is the name of the tool
    # the tool call response is the content of the message, overriding the existing content
    # (the only content before this should be the tool call)
    if message.additional_kwargs.get("tool_call_id"):
        function_response_part = types.Part.from_function_response(
            name=message.additional_kwargs.get("tool_call_id"),
            response={"result": message.content},
        )
        return types.Content(
            role=ROLES_TO_GEMINI[message.role], parts=[function_response_part]
        )

    return types.Content(
        role=ROLES_TO_GEMINI[message.role],
        parts=parts,
    )


def convert_schema_to_function_declaration(
    client: google.genai.client, tool: "BaseTool"
):
    if not tool.metadata.fn_schema:
        raise ValueError("fn_schema is missing")

    # Get the JSON schema
    root_schema = _transformers.t_schema(client, tool.metadata.fn_schema)

    description_parts = tool.metadata.description.split("\n", maxsplit=1)
    if len(description_parts) > 1:
        description = description_parts[-1]
    elif len(description_parts) == 1:
        description = description_parts[0]
    else:
        description = None

    # Create the function declaration
    return types.FunctionDeclaration(
        description=description,
        name=tool.metadata.name,
        parameters=root_schema,
    )


class ChatParams(typing.TypedDict):
    model: str
    history: list[types.Content]
    config: types.GenerateContentConfig


def prepare_chat_params(
    model: str, messages: Sequence[ChatMessage], **kwargs: Any
) -> tuple[types.Content, ChatParams]:
    """
    Prepare common parameters for chat creation.

    Args:
        messages: Sequence of chat messages
        **kwargs: Additional keyword arguments

    Returns:
        tuple containing:
        - next_msg: the next message to send
        - chat_kwargs: processed keyword arguments for chat creation

    """
    merged_messages = merge_neighboring_same_role_messages(messages)
    initial_history = list(map(chat_message_to_gemini, merged_messages))

    # merge tool messages into a single tool message
    # while maintaining the tool names
    history = []
    for idx, msg in enumerate(initial_history):
        if idx < 1:
            history.append(msg)
            continue

        # Skip if the role is different or not a tool message
        if msg.parts and not any(
            part.function_response is not None for part in msg.parts
        ):
            history.append(msg)
            continue

        last_msg = history[-1]

        # Skip if the last message is not a tool message
        if last_msg.parts and not any(
            part.function_response is not None for part in last_msg.parts
        ):
            history.append(msg)
            continue

        # Skip if the role is different
        if last_msg.role != msg.role:
            history.append(msg)
            continue

        # Merge the tool messages
        last_msg.parts.extend(msg.parts or [])

    # Separate the next message from the history
    next_msg = history.pop()

    tools: types.Tool | list[types.Tool] | None = kwargs.pop("tools", None)
    if tools and not isinstance(tools, list):
        tools = [tools]

    config: Union[types.GenerateContentConfig, dict] = kwargs.pop(
        "generation_config", {}
    )
    if not isinstance(config, dict):
        config = config.model_dump()

    chat_kwargs: ChatParams = {"model": model, "history": history}

    if tools:
        if not config.get("automatic_function_calling"):
            config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True, maximum_remote_calls=None
            )

        if not config.get("tool_config"):
            config["tool_config"] = kwargs.pop("tool_config", None)

        if not config.get("tools"):
            config["tools"] = tools

    chat_kwargs["config"] = types.GenerateContentConfig(**config)

    return next_msg, chat_kwargs


def handle_streaming_flexible_model(
    current_json: str,
    candidate: types.Candidate,
    output_cls: Type[BaseModel],
    flexible_model: Type[BaseModel],
) -> Tuple[Optional[BaseModel], str]:
    parts = candidate.content.parts or []
    data = parts[0].text if parts else None
    if data:
        current_json += data
        try:
            return output_cls.model_validate_json(current_json), current_json
        except ValidationError:
            try:
                return flexible_model.model_validate_json(
                    _repair_incomplete_json(current_json)
                ), current_json
            except ValidationError:
                return None, current_json

    return None, current_json


def _should_retry(exception: BaseException):
    if isinstance(exception, errors.ClientError):
        if exception.status in (429, 408):
            return True
    return False


def create_retry_decorator(
    max_retries: int,
    random_exponential: bool = False,
    stop_after_delay_seconds: Optional[float] = None,
    min_seconds: float = 4,
    max_seconds: float = 60,
) -> typing.Callable[[Any], Any]:
    wait_strategy = (
        wait_random_exponential(min=min_seconds, max=max_seconds)
        if random_exponential
        else wait_exponential(multiplier=1, min=min_seconds, max=max_seconds)
    )

    stop_strategy: stop_base = stop_after_attempt(max_retries)
    if stop_after_delay_seconds is not None:
        stop_strategy = stop_strategy | stop_after_delay(stop_after_delay_seconds)

    return retry(
        reraise=True,
        stop=stop_strategy,
        wait=wait_strategy,
        retry=(
            retry_if_exception_type(
                (errors.ServerError, httpx.ConnectError, httpx.ConnectTimeout)
            )
            | retry_if_exception(_should_retry)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
