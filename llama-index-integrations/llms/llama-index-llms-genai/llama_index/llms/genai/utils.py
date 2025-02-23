from typing import (
    Any,
    Dict,
)

from google.genai import types
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ImageBlock,
    TextBlock,
)
from llama_index.core.utilities.gemini_utils import (
    ROLES_FROM_GEMINI,
    ROLES_TO_GEMINI,
)


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


def completion_from_gemini_response(
    response: types.GenerateContentResponse,
) -> CompletionResponse:
    if not response.candidates:
        raise ValueError("Response has no candidates")

    top_candidate = response.candidates[0]
    _error_if_finished_early(top_candidate)

    raw = {
        **(top_candidate.model_dump()),
        **(response.prompt_feedback.model_dump() if response.prompt_feedback else {}),
    }

    if response.usage_metadata:
        raw["usage_metadata"] = response.usage_metadata.model_dump()
    return CompletionResponse(text=response.text or "", raw=raw)


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

    try:
        text = response.text
    except ValueError:
        text = None

    additional_kwargs: Dict[str, Any] = {}

    if response.function_calls:
        for fn in response.function_calls:
            if "tool_calls" not in additional_kwargs:
                additional_kwargs["tool_calls"] = []
            additional_kwargs["tool_calls"].append(fn)

    role = ROLES_FROM_GEMINI[top_candidate.content.role]
    return ChatResponse(
        message=ChatMessage(
            role=role, content=text, additional_kwargs=additional_kwargs
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
        parts.append(tool_call)

    return types.Content(
        role=ROLES_TO_GEMINI[message.role],
        parts=parts,
    )
