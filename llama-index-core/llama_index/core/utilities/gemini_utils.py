"""Global Gemini Utilities (shared between Gemini LLM and Vertex)."""

from collections.abc import Sequence
from typing import Dict

from llama_index.core.base.llms.types import ChatMessage, MessageRole

ROLES_TO_GEMINI: Dict[MessageRole, MessageRole] = {
    MessageRole.USER: MessageRole.USER,
    MessageRole.ASSISTANT: MessageRole.MODEL,
    ## Gemini only has user and model roles. Put the rest in user role.
    MessageRole.SYSTEM: MessageRole.USER,
    MessageRole.MODEL: MessageRole.MODEL,
}
ROLES_FROM_GEMINI: Dict[MessageRole, MessageRole] = {
    ## Gemini only has user and model roles.
    MessageRole.USER: MessageRole.USER,
    MessageRole.MODEL: MessageRole.ASSISTANT,
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
            role=ROLES_TO_GEMINI[current_message.role],
            content="\n".join([str(msg_content) for msg_content in merged_content]),
            additional_kwargs=current_message.additional_kwargs,
        )
        merged_messages.append(merged_message)
        i += 1

    return merged_messages
