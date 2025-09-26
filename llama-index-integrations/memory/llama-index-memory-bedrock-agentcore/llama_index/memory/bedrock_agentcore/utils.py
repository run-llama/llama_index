from datetime import datetime
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

    return formatted_messages


def convert_message_to_event_payload(
    message: ChatMessage, eventTimestamp: datetime
) -> Dict[str, Any]:
    payload = []

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

    payload.append({"blob": json.dumps({"eventTimeStamp": eventTimestamp.isoformat()})})

    payload.append({"blob": json.dumps(message.additional_kwargs)})

    payload.append(
        {
            "conversational": {"content": {"text": text}, "role": role},
        }
    )

    return payload


def convert_events_to_messages(events):
    """
    An event is stored with the following payload structure:
    1. first item will be the exact eventTimeStamp. This is required because boto3 resolves timestamps at 1 second,
       so tool calls can overlap and are not guaranteed to be sorted in ListEvents calls.
    2. second item will be the 'additional_kwargs' which contain the tool call information from the agent
    3. third item will be the conversational data containing the role & content of the message.
    """
    messages_with_timestamps = []

    for event in events:
        payload = event["payload"]
        if not payload:
            continue

        eventTimestampString = json.loads(payload[0]["blob"])["eventTimeStamp"]
        eventTimestamp = datetime.fromisoformat(eventTimestampString)
        tool_call_kwargs = json.loads(payload[1]["blob"])
        event_role = payload[-1]["conversational"]["role"]
        block_content = payload[-1]["conversational"]["content"]["text"]
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

        # Store message along with its timestamp
        messages_with_timestamps.append(
            (
                eventTimestamp,
                ChatMessage(
                    role=role, content=block_content, additional_kwargs=tool_call_kwargs
                ),
            )
        )

    return messages_with_timestamps


def convert_messages_to_string(
    messages: List[ChatMessage], input: Optional[str] = None
) -> str:
    formatted_messages = [f"{msg.role.value}: {msg.content}" for msg in messages]
    result = "\n".join(formatted_messages)

    if input:
        result += f"\nuser: {input}"

    return result
