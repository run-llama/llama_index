from typing import Union

import google.ai.generativelanguage as glm
import google.generativeai as genai
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ImageBlock,
    TextBlock,
)
from llama_index.core.multi_modal_llms.base import ChatMessage
from llama_index.core.utilities.gemini_utils import ROLES_FROM_GEMINI, ROLES_TO_GEMINI


def _error_if_finished_early(candidate: "glm.Candidate") -> None:  # type: ignore[name-defined] # only until release
    if (finish_reason := candidate.finish_reason) > 1:  # 1=STOP (normally)
        reason = finish_reason.name

        # Safety reasons have more detail, so include that if we can.
        if finish_reason == 3:  # 3=Safety
            relevant_safety = list(
                filter(
                    lambda sr: sr.probability > 1,  # 1=Negligible
                    candidate.safety_ratings,
                )
            )
            reason += f" {relevant_safety}"

        raise RuntimeError(f"Response was terminated early: {reason}")


def completion_from_gemini_response(
    response: Union[
        "genai.types.GenerateContentResponse",
        "genai.types.AsyncGenerateContentResponse",
    ],
) -> CompletionResponse:
    top_candidate = response.candidates[0]
    _error_if_finished_early(top_candidate)

    raw = {
        **(type(top_candidate).to_dict(top_candidate)),  # type: ignore
        **(type(response.prompt_feedback).to_dict(response.prompt_feedback)),  # type: ignore
    }
    if response.usage_metadata:
        raw["usage_metadata"] = type(response.usage_metadata).to_dict(
            response.usage_metadata
        )
    return CompletionResponse(text=response.text, raw=raw)


def chat_from_gemini_response(
    response: Union[
        "genai.types.GenerateContentResponse",
        "genai.types.AsyncGenerateContentResponse",
    ],
) -> ChatResponse:
    top_candidate = response.candidates[0]
    _error_if_finished_early(top_candidate)

    raw = {
        **(type(top_candidate).to_dict(top_candidate)),  # type: ignore
        **(type(response.prompt_feedback).to_dict(response.prompt_feedback)),  # type: ignore
    }
    if response.usage_metadata:
        raw["usage_metadata"] = type(response.usage_metadata).to_dict(
            response.usage_metadata
        )
    role = ROLES_FROM_GEMINI[top_candidate.content.role]
    return ChatResponse(message=ChatMessage(role=role, content=response.text), raw=raw)


def chat_message_to_gemini(message: ChatMessage) -> "genai.types.ContentDict":
    """Convert ChatMessages to Gemini-specific history, including ImageDocuments."""
    parts = []
    content_txt = ""
    for block in message.blocks:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ImageBlock):
            base64_bytes = block.resolve_image(as_base64=False).read()
            parts.append(
                {
                    "mime_type": block.image_mimetype,
                    "data": base64_bytes,
                }
            )
        else:
            msg = f"Unsupported content block type: {type(block).__name__}"
            raise ValueError(msg)

    return {
        "role": ROLES_TO_GEMINI[message.role],
        "parts": parts,
    }
