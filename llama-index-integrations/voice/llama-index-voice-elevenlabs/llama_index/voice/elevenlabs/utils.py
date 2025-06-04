from typing import Any, Dict, Optional, List
from llama_index.core.llms import ChatMessage, AudioBlock, TextBlock, MessageRole


def callback_user_message(
    messages: Dict[int, List[ChatMessage]],
    message_id: int,
    text: Optional[str] = None,
    audio: Optional[Any] = None,
) -> None:
    if messages.get(message_id) is None:
        if text:
            messages[message_id] = []
            messages[message_id].append(
                ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=text)])
            )
        else:
            messages[message_id] = []
            messages[message_id].append(
                ChatMessage(role=MessageRole.USER, blocks=[AudioBlock(audio=audio)])
            )
    else:
        last_agent_message = [
            message
            for message in messages[message_id]
            if message.role == MessageRole.USER
        ][-1]
        if text:
            last_agent_message.blocks.append(TextBlock(text=text))
        else:
            last_agent_message.blocks.append(AudioBlock(audio=audio))


def callback_agent_message(
    messages: Dict[int, List[ChatMessage]],
    message_id: int,
    text: Optional[str] = None,
    audio: Optional[Any] = None,
) -> None:
    if messages.get(message_id) is None:
        if text:
            messages[message_id] = []
            messages[message_id].append(
                ChatMessage(role=MessageRole.ASSISTANT, blocks=[TextBlock(text=text)])
            )
        else:
            messages[message_id] = []
            messages[message_id].append(
                ChatMessage(
                    role=MessageRole.ASSISTANT, blocks=[AudioBlock(audio=audio)]
                )
            )
    else:
        last_agent_message = [
            message
            for message in messages[message_id]
            if message.role == MessageRole.ASSISTANT
        ][-1]
        if text:
            last_agent_message.blocks.append(TextBlock(text=text))
        else:
            last_agent_message.blocks.append(AudioBlock(audio=audio))


def callback_agent_message_correction(
    messages: Dict[int, List[ChatMessage]], message_id: int, text: str
) -> None:
    last_agent_message = [
        message
        for message in messages[message_id]
        if message.role == MessageRole.ASSISTANT
    ][-1]
    last_block = [
        block for block in last_agent_message.blocks if block.block_type == "text"
    ][-1]
    last_block.text = text


def callback_latency_measurement(latencies: List[int], latency: int) -> None:
    latencies.append(latency)
