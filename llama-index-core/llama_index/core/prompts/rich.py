import base64
from urllib.parse import urlparse

from banks import Prompt
from banks.types import ContentBlockType as BanksContentBlockType
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as default_messages_to_prompt,
)
from llama_index.core.base.llms.types import (
    ContentBlock,
    TextBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock,
    DocumentBlock,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.types import BaseOutputParser


def _is_url(data: str) -> bool:
    """Check whether a banks media payload is a URL rather than base64 data."""
    return bool(urlparse(data).scheme)


# banks DocumentFormat values mapped to mimetypes
_DOCUMENT_FORMAT_MIMETYPES = {"pdf": "application/pdf", "txt": "text/plain"}


class RichPromptTemplate(BasePromptTemplate):  # type: ignore[no-redef]
    template_str: str = Field(description="The template string for the prompt.")

    def __init__(
        self,
        template_str: str,
        metadata: Optional[Dict[str, Any]] = None,
        output_parser: Optional[BaseOutputParser] = None,
        template_vars: Optional[List[str]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
        **kwargs: Any,
    ):
        template_vars = template_vars or []
        if not template_vars:
            template_vars = Prompt(template_str).variables

        super().__init__(
            template_str=template_str,
            kwargs=kwargs or {},
            metadata=metadata or {},
            output_parser=output_parser,
            template_vars=template_vars,
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )

    @property
    def is_chat_template(self) -> bool:
        return "endchat" in self.template_str

    def partial_format(self, **kwargs: Any) -> "RichPromptTemplate":
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)
        return prompt

    def format(
        self,
        llm: Optional[BaseLLM] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        **kwargs: Any,
    ) -> str:
        del llm  # unused

        if self.is_chat_template:
            messages = self.format_messages(**kwargs)

            if messages_to_prompt is not None:
                return messages_to_prompt(messages)

            return default_messages_to_prompt(messages)
        else:
            all_kwargs = {
                **self.kwargs,
                **kwargs,
            }
            mapped_all_kwargs = self._map_all_vars(all_kwargs)
            return Prompt(self.template_str).text(data=mapped_all_kwargs)

    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        del llm  # unused
        """Format the prompt into a list of chat messages."""
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }
        mapped_all_kwargs = self._map_all_vars(all_kwargs)

        banks_prompt = Prompt(self.template_str)
        banks_messages = banks_prompt.chat_messages(data=mapped_all_kwargs)

        llama_messages: list[ChatMessage] = []
        for bank_message in banks_messages:
            if isinstance(bank_message.content, str):
                llama_messages.append(
                    ChatMessage(role=bank_message.role, content=bank_message.content)
                )
            elif isinstance(bank_message.content, list):
                llama_blocks: list[ContentBlock] = []
                for bank_block in bank_message.content:
                    if bank_block.type == BanksContentBlockType.text:
                        llama_blocks.append(TextBlock(text=bank_block.text))
                    elif bank_block.type == BanksContentBlockType.image_url:
                        llama_blocks.append(ImageBlock(url=bank_block.image_url.url))
                    elif bank_block.type == BanksContentBlockType.audio:
                        input_audio = bank_block.input_audio
                        if _is_url(input_audio.data):
                            llama_blocks.append(AudioBlock(url=input_audio.data))
                        else:
                            llama_blocks.append(
                                AudioBlock(
                                    audio=base64.b64decode(input_audio.data),
                                    format=input_audio.format,
                                )
                            )
                    elif bank_block.type == BanksContentBlockType.video:
                        input_video = bank_block.input_video
                        if _is_url(input_video.data):
                            llama_blocks.append(VideoBlock(url=input_video.data))
                        else:
                            llama_blocks.append(
                                VideoBlock(
                                    video=base64.b64decode(input_video.data),
                                    video_mimetype=f"video/{input_video.format}",
                                )
                            )
                    elif bank_block.type == BanksContentBlockType.document:
                        input_document = bank_block.input_document
                        if _is_url(input_document.data):
                            llama_blocks.append(DocumentBlock(url=input_document.data))
                        else:
                            llama_blocks.append(
                                DocumentBlock(
                                    data=base64.b64decode(input_document.data),
                                    document_mimetype=_DOCUMENT_FORMAT_MIMETYPES.get(
                                        input_document.format
                                    ),
                                )
                            )
                    else:
                        raise ValueError(
                            f"Unsupported content block type: {bank_block.type}"
                        )

                llama_messages.append(
                    ChatMessage(role=bank_message.role, content=llama_blocks)
                )
            else:
                raise ValueError(
                    f"Unsupported message content type: {type(bank_message.content)}"
                )

        if self.output_parser is not None:
            llama_messages = self.output_parser.format_messages(llama_messages)

        return llama_messages

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        return self.template_str
