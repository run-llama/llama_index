"""Epub parser.

Contains parsers for epub files.
"""

from pathlib import Path
from typing import Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class EpubReader(BaseReader):
    """Epub Parser."""

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            raise ImportError(
                "`EbookLib` is required to read Epub files: `pip install EbookLib`"
            )
        try:
            import html2text
        except ImportError:
            raise ImportError(
                "`html2text` is required to parse Epub files: `pip install html2text`"
            )

        text_list = []
        book = epub.read_epub(file, options={"ignore_ncx": True})

        # Iterate through all chapters.
        for item in book.get_items():
            # Chapters are typically located in epub documents items.
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                text_list.append(
                    html2text.html2text(item.get_content().decode("utf-8"))
                )

        text = "\n".join(text_list)
        return [Document(text, extra_info=extra_info)]
