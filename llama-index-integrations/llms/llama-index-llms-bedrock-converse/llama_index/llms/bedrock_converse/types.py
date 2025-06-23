from llama_index.core.bridge.pydantic import Field
from llama_index.core.base.llms.types import TextBlock


class BedrockConverseTextBlock(TextBlock):
    additional_kwargs: dict = Field(default_factory=dict)
