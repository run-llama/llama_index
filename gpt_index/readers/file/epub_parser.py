"""Epub parser.

Contains parsers for epub files.
"""

from pathlib import Path
from typing import Dict

from gpt_index.readers.file.base_parser import BaseParser


class EpubParser(BaseParser):
    """Epub Parser."""

    def _init_parser(self) -> Dict:
        """Init parser."""
        return {}

    def parse_file(self, file: Path, errors: str = "ignore") -> str:
        """Parse file."""
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            raise ValueError("`EbookLib` is required to read Epub files.")
        try:
            import html2text
        except ImportError:
            raise ValueError("`html2text` is required to parse Epub files.")

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
        return text
