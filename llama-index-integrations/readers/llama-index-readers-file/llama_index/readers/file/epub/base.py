"""
Epub parser.

Contains parsers for epub files.
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging
from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class EpubReader(BaseReader):
    """Epub Parser."""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        try:
            import ebooklib
            import html2text
            from ebooklib import epub
        except ImportError:
            raise ImportError(
                "Please install extra dependencies that are required for "
                "the EpubReader: "
                "`pip install EbookLib html2text`"
            )
        if fs:
            logger.warning(
                "fs was specified but EpubReader doesn't support loading "
                "from fsspec filesystems. Will load from local filesystem instead."
            )

        book = epub.read_epub(file, options={"ignore_ncx": True})

        # Chapters are typically located in epub documents items. Their reading
        # order is the spine; the manifest lists them in no particular order.
        # Documents the spine leaves out follow it, in manifest order.
        documents = [
            item
            for item in book.get_items()
            if item.get_type() == ebooklib.ITEM_DOCUMENT
        ]
        ordered = []
        for idref, _linear in book.spine:
            item = book.get_item_with_id(idref)
            if item in documents and item not in ordered:
                ordered.append(item)
        ordered.extend(item for item in documents if item not in ordered)

        text_list = [
            html2text.html2text(item.get_content().decode("utf-8")) for item in ordered
        ]
        text = "\n".join(text_list)
        return [Document(text=text, metadata=extra_info or {})]
