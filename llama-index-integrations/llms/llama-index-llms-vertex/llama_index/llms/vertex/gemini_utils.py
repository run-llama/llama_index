import base64
from typing import Any, Dict, Union, Optional
from vertexai.generative_models._generative_models import SafetySettingsType
from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types
from llama_index.core.llms import ChatMessage, MessageRole


def is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


def create_gemini_client(
    model: str, safety_settings: Optional[SafetySettingsType]
) -> Any:
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

    if message.content == "" and "tool_calls" in message.additional_kwargs:
        tool_calls = message.additional_kwargs["tool_calls"]
        parts = [
            Part._from_gapic(raw_part=gapic_content_types.Part(function_call=tool_call))
            for tool_call in tool_calls
        ]
    else:
        raw_content = message.content

        if raw_content is None:
            raw_content = ""
        if isinstance(raw_content, str):
            raw_content = [raw_content]

        parts = [_convert_gemini_part_to_prompt(part) for part in raw_content]

    if is_history:
        return Content(
            role="user" if message.role == MessageRole.USER else "model",
            parts=parts,
        )
    else:
        return parts
