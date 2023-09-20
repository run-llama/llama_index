"""Markdown splitter."""
from typing import List, Optional
from llama_index.bridge.pydantic import Field

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.text_splitter.types import TextSplitter
import re


class MarkdownSplitter(TextSplitter):
    """Implementation of splitting text for Markdown files."""

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
    ):
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MarkdownSplitter"

    def split_text(self, text: str) -> List[str]:
        if text == "":
            return []

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ):
            return self._split_text(text)

    def _split_text(self, text: str) -> List[str]:
        markdown_sections = []
        lines = text.split("\n")

        current_section = ""

        for line in lines:
            header_match = re.match(r"^(#+)\s(.*)", line)
            if header_match:
                if current_section != "":
                    markdown_sections.append(current_section.strip())
                current_section = f"{header_match.group(2)}\n"
            else:
                current_section += line + "\n"

        markdown_sections.append(current_section.strip())

        return markdown_sections
