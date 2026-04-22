import logging
import json
from typing import Any, Dict, List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

DEFAULT_INTRO_PREFERENCES = "Below are a set of relevant preferences retrieved from potentially several memory sources:"
DEFAULT_OUTRO_PREFERENCES = "This is the end of the retrieved preferences."
DISCLAIMER_FOR_LLM = "IMPORTANT: Ignore preferences unless they conflict. Proceed to answer the following user query directly and use tools if appropriate."
# For tool calls, there is a corresponding Assistant message that has an empty text. It's needed to reconstruct the entire conversation; however,
# CreateEvent doesn't accept empty text payloads, so this placeholder is needed. We will strip it out during ListEvents so that we don't influence the Agent with random text.
EMPTY_PAYLOAD_PLACEHOLDER_TEXT = "PLACEHOLDER FOR EMPTY ASSISTANT"


def convert_memory_to_system_message(
    memory_records: List[Dict[str, Any]], existing_system_message: ChatMessage = None
) -> ChatMessage:
    memories = [memory_json.get("text", "") for memory_json in memory_records]
    formatted_messages = "\n\n" + DEFAULT_INTRO_PREFERENCES + "\n"
    for memory in memories:
        formatted_messages += f"\n {memory} \n\n"
    formatted_messages += DEFAULT_OUTRO_PREFERENCES
    system_message = formatted_messages
    # If existing system message is available
    if existing_system_message is not None:
        system_message = existing_system_message.content.split(
            DEFAULT_INTRO_PREFERENCES
        )[0]
        system_message = system_message + formatted_messages
    return ChatMessage(content=system_message, role=MessageRole.SYSTEM)


def convert_memory_to_user_message(memory_records: List[Dict[str, Any]]) -> str:
    memories = [memory_json.get("text", "") for memory_json in memory_records]
    formatted_messages = "\n\n" + DEFAULT_INTRO_PREFERENCES + "\n"
    for memory in memories:
        formatted_messages += f"\n {memory} \n\n"

    formatted_messages += f"{DEFAULT_OUTRO_PREFERENCES}\n"
    formatted_messages += f"{DISCLAIMER_FOR_LLM}\n"

    return ChatMessage(content=formatted_messages, role=MessageRole.USER)


def convert_messages_to_event_payload(messages: List[ChatMessage]) -> Dict[str, Any]:
    payload = []

    for message in messages:
        text = message.content
        if not text.strip():
            text = EMPTY_PAYLOAD_PLACEHOLDER_TEXT

        # Map LangChain roles to Bedrock Agent Core roles
        if message.role == MessageRole.USER:
            role = "USER"
        elif message.role == MessageRole.ASSISTANT:
            role = "ASSISTANT"
        elif message.role == MessageRole.TOOL:
            role = "TOOL"
        elif message.role == MessageRole.SYSTEM:
            role = "OTHER"
        else:
            logger.warning(f"Skipping unsupported message type: {message.role}")
            return None

        # payload.append({"blob": json.dumps({"eventTimeStamp": eventTimestamp.isoformat()})})

        payload.append({"blob": json.dumps(message.additional_kwargs)})

        payload.append(
            {
                "conversational": {"content": {"text": text}, "role": role},
            }
        )

    return payload


def convert_events_to_messages(events):
    """
    Reconstruct chat messages from event payloads.
    Each message consists of:
    1. a 'blob' entry containing tool call kwargs (JSON),
    2. followed by a 'conversational' entry containing role + text.
    """
    messages = []

    for event in events:
        payload = event.get("payload")
        if not payload:
            continue

        # walk the payload in pairs (blob, conversational)
        for i in range(0, len(payload), 2):
            blob_entry = payload[i].get("blob")
            conv_entry = payload[i + 1].get("conversational")

            tool_call_kwargs = json.loads(blob_entry) if blob_entry else {}
            event_role = conv_entry["role"]
            block_content = conv_entry["content"]["text"]

            if block_content == EMPTY_PAYLOAD_PLACEHOLDER_TEXT:
                block_content = ""

            if event_role == "USER":
                role = MessageRole.USER
            elif event_role == "ASSISTANT":
                role = MessageRole.ASSISTANT
            elif event_role == "TOOL":
                role = MessageRole.TOOL
            elif event_role == "OTHER":
                role = MessageRole.SYSTEM
            else:
                logger.warning(f"Skipping unsupported event role type: {event_role}")
                continue

            messages.append(
                ChatMessage(
                    role=role,
                    content=block_content,
                    additional_kwargs=tool_call_kwargs,
                )
            )

    return messages


def convert_messages_to_string(
    messages: List[ChatMessage], input: Optional[str] = None
) -> str:
    formatted_messages = [f"{msg.role.value}: {msg.content}" for msg in messages]
    result = "\n".join(formatted_messages)

    if input:
        result += f"\nuser: {input}"

    return result
