from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


class FunctionMessage(BaseMessage):
    """Type of message that is spoken by the AI."""

    name: str

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "function"


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    if isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "name": message.name,
            "content": message.content,
        }
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def monkey_patch_langchain() -> None:
    from langchain import schema
    from langchain.chat_models import openai

    # monkey patch
    openai._convert_message_to_dict = _convert_message_to_dict
    schema.FunctionMessage = FunctionMessage  # type: ignore
