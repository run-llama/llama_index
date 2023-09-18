"""HTML splitter."""
from typing import List, Optional
from llama_index.bridge.pydantic import Field

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.text_splitter.types import TextSplitter


class HTMLSplitter(TextSplitter):
    from bs4 import Tag

    """Implementation of splitting text for HTML files."""

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )
    tags: List[str] = Field(description="The HTML tags to extract.")

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        tags: Optional[List[str]] = ["section"],
    ):
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            callback_manager=callback_manager,
            tags=tags,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "HTMLSplitter"

    def split_text(self, text: str) -> List[str]:
        if text == "":
            return []

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ):
            return self._split_text(text)

    def _split_text(self, text: str) -> List[str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("bs4 is required to read HTML files.")

        soup = BeautifulSoup(text, "html.parser")
        splits = []
        tags = soup.find_all(self.tags)
        for tag in tags:
            tag_text = self._extract_text_from_tag(tag)
            splits.append(tag_text)

        return splits

    def _extract_text_from_tag(self, tag: Tag) -> str:
        from bs4 import NavigableString

        texts = []
        for elem in tag.children:
            if isinstance(elem, NavigableString):
                if elem.strip():
                    texts.append(elem.strip())
            elif elem.name == self._tag:
                continue
            else:
                texts.append(elem.get_text().strip())
        return "\n".join(texts)
