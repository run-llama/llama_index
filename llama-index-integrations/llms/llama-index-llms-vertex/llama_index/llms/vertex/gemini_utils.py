from typing import Any, Union, Optional
from vertexai.generative_models._generative_models import SafetySettingsType
from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types
from llama_index.core.llms import ChatMessage, MessageRole, ImageBlock, TextBlock


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

    def _convert_block_to_part(block: Union[TextBlock, ImageBlock]) -> Optional[Part]:
        from vertexai.preview.generative_models import Image

        if isinstance(block, TextBlock):
            if block.text:
                return Part.from_text(block.text)
            else:
                return None
        elif isinstance(block, ImageBlock):
            if block.path:
                image = Image.load_from_file(block.path)
            elif block.image:
                image = Image.from_bytes(block.image)
            elif block.url:
                base64_bytes = block.resolve_image(as_base64=False).read()
                image = Image.from_bytes(base64_bytes)
            else:
                raise ValueError("ImageBlock must have either path, url, or image data")
            return Part.from_image(image)
        else:
            raise ValueError(f"Unsupported block type: {type(block).__name__}")

    if (
        message.content == "" or message.content is None
    ) and "tool_calls" in message.additional_kwargs:
        tool_calls = message.additional_kwargs["tool_calls"]
        parts = [
            Part._from_gapic(raw_part=gapic_content_types.Part(function_call=tool_call))
            for tool_call in tool_calls
        ]
    else:
        if not isinstance(message.blocks, list):
            raise ValueError("message.blocks must be a list of content blocks")
        parts = [
            part
            for part in (_convert_block_to_part(block) for block in message.blocks)
            if part is not None
        ]

    if is_history:
        return Content(
            role="user" if message.role == MessageRole.USER else "model",
            parts=parts,
        )
    else:
        return parts
