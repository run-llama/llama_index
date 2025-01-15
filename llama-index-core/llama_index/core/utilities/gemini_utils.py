"""Global Gemini Utilities (shared between Gemini LLM and Vertex)."""

from __future__ import annotations

from collections.abc import Sequence

from llama_index.core.base.llms.types import ChatMessage, MessageRole

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
