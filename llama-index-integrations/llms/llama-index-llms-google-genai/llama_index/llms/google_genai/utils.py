from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Union,
)
import typing

import google.genai.types as types
import google.genai
from google.genai import _transformers
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ImageBlock,
    TextBlock,
)
from llama_index.core.utilities.gemini_utils import (
    ROLES_FROM_GEMINI,
    ROLES_TO_GEMINI,
    merge_neighboring_same_role_messages,
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


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
    json_schema = tool.metadata.fn_schema.model_json_schema()

    if json_schema.get("properties"):
        _transformers.process_schema(json_schema, client)
        root_schema = json_schema
    else:
        root_schema = None

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
    *history, next_msg = map(chat_message_to_gemini, merged_messages)

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
