from typing import Optional

from llama_index.core.base.llms.types import ChatMessage
from typing_extensions import NotRequired, TypedDict

XINFERENCE_MODEL_SIZES = {
    "baichuan": 2048,
    "baichuan-chat": 2048,
    "wizardlm-v1.0": 2048,
    "vicuna-v1.3": 2048,
    "orca": 2048,
    "chatglm": 2048,
    "chatglm2": 8192,
    "llama-2-chat": 4096,
    "llama-2": 4096,
}


class ChatCompletionMessage(TypedDict):
    role: str
    content: Optional[str]
    user: NotRequired[str]


def xinference_message_to_history(message: ChatMessage) -> ChatCompletionMessage:
    return ChatCompletionMessage(role=message.role, content=message.content)


def xinference_modelname_to_contextsize(modelname: str) -> int:
    context_size = XINFERENCE_MODEL_SIZES.get(modelname)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(XINFERENCE_MODEL_SIZES.keys())
        )

    return context_size
