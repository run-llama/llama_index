from typing import Any, Dict, Sequence, Tuple

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole


HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"

FUNCTION_CALLING_MODELS = {
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "anthropic.claude-3-opus-20240229-v1:0": 200000,
    "cohere.command-r-plus-v1:0": 128000,
    "mistral.mistral-large-2402-v1:0": 32000,
}


def __merge_common_role_msgs(
    messages: Sequence[Dict[str, Any]],
) -> Sequence[Dict[str, Any]]:
    """Merge consecutive messages with the same role."""
    postprocessed_messages: Sequence[Dict[str, Any]] = []
    for message in messages:
        if (
            postprocessed_messages
            and postprocessed_messages[-1]["role"] == message["role"]
        ):
            postprocessed_messages[-1]["content"] += message["content"]
        else:
            postprocessed_messages.append(message)
    return postprocessed_messages


def messages_to_converse_messages(
    messages: Sequence[ChatMessage],
) -> Tuple[Sequence[Dict[str, Any]], str]:
    """
    Converts a list of generic ChatMessages to AWS Bedrock Converse messages.

    Args:
        messages: List of ChatMessages

    Returns:
        Tuple of:
        - List of AWS Bedrock Converse messages
        - System prompt
    """
    converse_messages = []
    system_prompt = ""
    for message in messages:
        if message.role == MessageRole.SYSTEM:
            # get the system prompt
            system_prompt += message.content + "\n"
        elif message.role == MessageRole.FUNCTION or message.role == MessageRole.TOOL:
            # convert tool output to the AWS Bedrock Converse format
            content = {
                "toolResult": {
                    "toolUseId": message.additional_kwargs["tool_call_id"][-1],
                    "content": [
                        {
                            "text": message.content,
                        },
                    ],
                    "status": message.additional_kwargs["status"][-1],
                }
            }
            converse_message = {
                "role": message.role.value,
                "content": content,
            }
            converse_messages.append(converse_message)
        else:
            content = []
            if message.content:
                # get the text of the message
                content.append({"text": message.content})
            # convert tool calls to the AWS Bedrock Converse format
            tool_calls = message.additional_kwargs.get("tool_calls", [])
            for tool_call in tool_calls:
                assert "toolUseId" in tool_call
                assert "input" in tool_call
                assert "name" in tool_call
                content.append(tool_call)
            converse_message = {
                "role": message.role.value,
                "content": content,
            }
            converse_messages.append(converse_message)

    return __merge_common_role_msgs(converse_messages), system_prompt.strip()


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]
