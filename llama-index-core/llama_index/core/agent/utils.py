"""Agent utils."""

from llama_index.core.llms import ChatMessage, TextBlock
from typing import List
from llama_index.core.agent.types import TaskStep
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import BaseMemory


def add_user_step_to_memory(
    step: TaskStep, memory: BaseMemory, verbose: bool = False
) -> None:
    """Add user step to memory."""
    user_message = ChatMessage(content=step.input, role=MessageRole.USER)
    memory.put(user_message)
    if verbose:
        print(f"Added user message to memory: {step.input}")


def messages_to_xml_format(messages: List[ChatMessage]) -> ChatMessage:
    blocks = [TextBlock(text="<current_conversation>\n")]
    for message in messages:
        blocks.append(TextBlock(text=f"\t<{message.role.value}>\n"))
        for block in message.blocks:
            if isinstance(block, TextBlock):
                blocks.append(TextBlock(text=f"\t\t<message>{block.text}</message>\n"))
        blocks.append(TextBlock(text=f"\t</{message.role.value}>\n"))
    blocks.append(TextBlock(text="</current_conversation>\n"))
    blocks.append(
        TextBlock(
            text="Given the current conversation, can you please format your output?"
        )
    )
    return ChatMessage(role="user", blocks=blocks)
