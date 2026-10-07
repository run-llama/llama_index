"""Agent utils."""

import json

from llama_index.core.llms import ChatMessage, TextBlock
from typing import List, Type, Dict, Any, Optional, cast
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM


def messages_to_xml_format(messages: List[ChatMessage]) -> List[ChatMessage]:
    blocks = [TextBlock(text="<current_conversation>\n")]
    system_msg: Optional[ChatMessage] = None
    for message in messages:
        if message.role.value == "system":
            system_msg = message
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
    messages = [ChatMessage(role="user", blocks=blocks)]
    if system_msg:
        messages.insert(0, system_msg)
    return messages


def _structured_content_to_dict(content: Any, raw: Any = None) -> Dict[str, Any]:
    """Turn structured-LLM chat content into a dict without assuming it is JSON text."""
    if isinstance(content, dict):
        return content
    if isinstance(content, BaseModel):
        return content.model_dump()
    if content is None:
        if isinstance(raw, BaseModel):
            return raw.model_dump()
        if isinstance(raw, dict):
            return raw
        return {}
    return cast(Dict[str, Any], json.loads(content))


async def generate_structured_response(
    messages: List[ChatMessage], llm: LLM, output_cls: Type[BaseModel]
) -> Dict[str, Any]:
    xml_message = messages_to_xml_format(messages)
    structured_response = await llm.as_structured_llm(
        output_cls,
    ).achat(messages=xml_message)
    return _structured_content_to_dict(
        structured_response.message.content, structured_response.raw
    )
