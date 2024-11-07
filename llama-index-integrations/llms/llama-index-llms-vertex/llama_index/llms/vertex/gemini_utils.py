import base64
from typing import Any, Dict, Union, Optional
from vertexai.generative_models import SafetySetting
from llama_index.core.llms import ChatMessage, MessageRole


def is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


def create_gemini_client(model: str, safety_settings: Optional[SafetySetting]) -> Any:
    from vertexai.preview.generative_models import GenerativeModel

    return GenerativeModel(model_name=model, safety_settings=safety_settings)


def convert_chat_message_to_gemini_content(
    message: ChatMessage, is_history: bool = True
) -> Any:
    from vertexai.preview.generative_models import Content, Part

    def _convert_gemini_part_to_prompt(part: Union[str, Dict]) -> Part:
        from vertexai.preview.generative_models import Image, Part

        if isinstance(part, str):
            return Part.from_text(part)

        if not isinstance(part, Dict):
            raise ValueError(
                f"Message's content is expected to be a dict, got {type(part)}!"
            )
        if part["type"] == "text":
            return Part.from_text(part["text"])
        elif part["type"] == "image_url":
            path = part["image_url"]
            if path.startswith("gs://"):
                raise ValueError("Only local image path is supported!")
            elif path.startswith("data:image/jpeg;base64,"):
                image = Image.from_bytes(base64.b64decode(path[23:]))
            else:
                image = Image.load_from_file(path)
        else:
            raise ValueError("Only text and image_url types are supported!")
        return Part.from_image(image)

    if (
        message.role == MessageRole.ASSISTANT
        and message.content == ""
        and "tool_calls" in message.additional_kwargs
    ) or (message.role == MessageRole.TOOL):
        tool_calls = message.additional_kwargs["tool_calls"]
        if message.role != MessageRole.TOOL:
            parts = [
                Part.from_function_response(tool_call.name, tool_call.args)
                for tool_call in tool_calls
            ]
            parts.append(Part.from_text(handle_raw_content(message)))
        else:
            ## this handles the case where the Gemini api properly sets the message role to tool instead of assistant
            if "name" in message.additional_kwargs:
                parts = [
                    Part.from_function_response(
                        message.additional_kwargs["name"],
                        message.additional_kwargs.get("args", {}),
                    )
                ]
            else:
                raise ValueError("Tool name must be provided!")

            raw_content = handle_raw_content(message)
            parts.append(_convert_gemini_part_to_prompt(part) for part in raw_content)
    else:
        raw_content = handle_raw_content(message)

        parts = [_convert_gemini_part_to_prompt(part) for part in raw_content]

    if is_history:
        return Content(
            role="user" if message.role == MessageRole.USER else "model",
            parts=parts,
        )
    else:
        return parts


def handle_raw_content(message):
    raw_content = message.content
    if raw_content is None:
        raw_content = ""
    if isinstance(raw_content, str):
        raw_content = [raw_content]
    return raw_content
