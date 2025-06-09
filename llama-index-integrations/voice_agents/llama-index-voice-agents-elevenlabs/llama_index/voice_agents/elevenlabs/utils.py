from pydantic import BaseModel
from inspect import Signature, Parameter

from typing import Any, Dict, Optional, List, Callable
from llama_index.core.llms import ChatMessage, AudioBlock, TextBlock, MessageRole
from llama_index.core.tools import BaseTool


def make_function_from_tool_model(
    model_cls: type[BaseModel], tool: BaseTool
) -> Callable:
    fields = model_cls.model_fields
    parameters = [
        Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, annotation=field.annotation)
        for name, field in fields.items()
    ]
    sig = Signature(parameters)

    def func_template(*args, **kwargs):
        bound = func_template.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        return tool(**bound.arguments).raw_output

    func_template.__signature__ = sig
    return func_template


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
        last_user_messages = [
            message
            for message in messages[message_id]
            if message.role == MessageRole.USER
        ]
        if len(last_user_messages) > 0:
            last_user_message = last_user_messages[-1]
        else:
            messages[message_id].append(ChatMessage(role=MessageRole.USER, blocks=[]))
            last_user_message = [
                message
                for message in messages[message_id]
                if message.role == MessageRole.USER
            ][-1]
        if text:
            last_user_message.blocks.append(TextBlock(text=text))
        else:
            last_user_message.blocks.append(AudioBlock(audio=audio))


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
        last_agent_messages = [
            message
            for message in messages[message_id]
            if message.role == MessageRole.ASSISTANT
        ]
        if len(last_agent_messages) > 0:
            last_agent_message = last_agent_messages[-1]
        else:
            messages[message_id].append(
                ChatMessage(role=MessageRole.ASSISTANT, blocks=[])
            )
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
