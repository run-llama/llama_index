from typing import Sequence
from llama_index.core.base.llms.types import ChatMessage, MessageRole


def prepare_messages_before_chat(messages: Sequence[ChatMessage], **all_kwargs):
    chat_messages = []

    for message in messages:
        if "system_prompt" in all_kwargs and message.role.value == MessageRole.SYSTEM:
            continue
        elif (
            "system_prompt" not in all_kwargs
            and message.role.value == MessageRole.SYSTEM
        ):
            all_kwargs["system_prompt"] = message.content
        elif message.role.value == MessageRole.ASSISTANT:
            chat_messages.append(
                {"role": message.role.value, "content": message.content}
            )
        elif message.role.value == MessageRole.USER:
            if "template_id" not in all_kwargs:
                chat_messages.append(
                    {"role": message.role.value, "content": message.content}
                )
            else:
                template_id = all_kwargs["template_id"]
                assert template_id is not None and template_id != "", ValueError(
                    "template_id can not be None or '' when passed in kwargs"
                )
                assert "id" in message.additional_kwargs, KeyError(
                    "When using Prem templates in llama-index, ensure you have 'id' key ",
                    "in message.additional_kwargs. This id act as the template variable. ",
                    "If you do not have template variable then use only system prompt",
                )

                chat_messages.append(
                    {
                        "role": message.role.value,
                        "template_id": template_id,
                        "params": {message.additional_kwargs["id"]: message.content},
                    }
                )

        else:
            raise ValueError("role can be either 'system', 'user' or 'assistant'")
    return chat_messages, all_kwargs
