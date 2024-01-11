import typing
from typing import Sequence, Union

from llama_index.core.llms.types import MessageRole
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)

if typing.TYPE_CHECKING:
    import google.ai.generativelanguage as glm
    import google.generativeai as genai


ROLES_TO_GEMINI = {
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "model",
    ## Gemini only has user and model roles. Put the rest in user role.
    MessageRole.SYSTEM: "user",
}
ROLES_FROM_GEMINI = {v: k for k, v in ROLES_TO_GEMINI.items()}


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
        **(type(top_candidate).to_dict(top_candidate)),
        **(type(response.prompt_feedback).to_dict(response.prompt_feedback)),
    }
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
        **(type(top_candidate).to_dict(top_candidate)),
        **(type(response.prompt_feedback).to_dict(response.prompt_feedback)),
    }
    role = ROLES_FROM_GEMINI[top_candidate.content.role]
    return ChatResponse(message=ChatMessage(role=role, content=response.text), raw=raw)


def chat_message_to_gemini(message: ChatMessage) -> "genai.types.ContentDict":
    """Convert ChatMessages to Gemini-specific history, including ImageDocuments."""
    parts = [message.content]
    if images := message.additional_kwargs.get("images"):
        try:
            import PIL

            parts += [PIL.Image.open(doc.resolve_image()) for doc in images]
        except ImportError:
            # This should have been caught earlier, but tell the user anyway.
            raise ValueError("Multi-modal support requires PIL.")

    return {
        "role": ROLES_TO_GEMINI[message.role],
        "parts": parts,
    }


def merge_neighboring_same_role_messages(
    messages: Sequence[ChatMessage],
) -> Sequence[ChatMessage]:
    # Gemini does not support multiple messages of the same role in a row, so we merge them
    merged_messages = []
    i = 0

    while i < len(messages):
        current_message = messages[i]
        # Initialize merged content with current message content
        merged_content = [current_message.content]

        # Check if the next message exists and has the same role
        while (
            i + 1 < len(messages)
            and ROLES_TO_GEMINI[messages[i + 1].role]
            == ROLES_TO_GEMINI[current_message.role]
        ):
            i += 1
            next_message = messages[i]
            merged_content.extend([next_message.content])

        # Create a new ChatMessage or similar object with merged content
        merged_message = ChatMessage(
            role=current_message.role,
            content="\n".join([str(msg_content) for msg_content in merged_content]),
            additional_kwargs=current_message.additional_kwargs,
        )
        merged_messages.append(merged_message)
        i += 1

    return merged_messages
