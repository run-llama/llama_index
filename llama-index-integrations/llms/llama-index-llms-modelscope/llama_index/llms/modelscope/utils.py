from typing import Dict, List, Sequence

from llama_index.core.base.llms.types import (
    ChatResponse,
    CompletionResponse,
    ChatMessage,
)


def chat_message_to_modelscope_messages(
    chat_messages: Sequence[ChatMessage],
) -> List[Dict]:
    messages = []
    for msg in chat_messages:
        messages.append({"role": msg.role.value, "content": msg.content})
    return {"messages": messages}


def text_to_completion_response(output) -> CompletionResponse:
    return CompletionResponse(text=output["text"], raw=output)


def modelscope_message_to_chat_response(output) -> ChatResponse:
    # output format: {'message': {'role': 'assistant', 'content': ''}}
    return ChatResponse(
        message=ChatMessage(
            role=output["message"]["role"], content=output["message"]["content"]
        ),
        raw=output,
    )
