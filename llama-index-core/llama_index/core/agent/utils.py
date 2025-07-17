"""Agent utils."""

import json

from llama_index.core.llms import ChatMessage, TextBlock
from typing import List, Type, Dict, Any, cast
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.agent.types import TaskStep
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
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
    blocks.append(TextBlock(text="</current_conversation>\n\n"))
    blocks.append(
        TextBlock(
            text="Given the conversation, format the output according to the provided schema."
        )
    )
    return ChatMessage(role="user", blocks=blocks)


async def generate_structured_response(
    messages: List[ChatMessage], llm: LLM, output_cls: Type[BaseModel]
) -> Dict[str, Any]:
    xml_message = messages_to_xml_format(messages)
    structured_response = await llm.as_structured_llm(
        output_cls,
    ).achat(messages=[xml_message])
    return cast(Dict[str, Any], json.loads(structured_response.message.content))
