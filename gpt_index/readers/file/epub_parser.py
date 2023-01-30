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

    def check_tag(self, text, blacklist=['[document]']):
        return text.parent.name not in blacklist

    def parse_file(self, file: Path, errors: str = "ignore") -> str:
        """Parse file."""
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            raise ValueError("`EbookLib` is required to read Epub files.")
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ValueError("`beautifulsoup4` is required to parse Epub files.")

        text_list = []
        book = epub.read_epub(file, options={'ignore_ncx': True})

        # Iterate through all chapters.
        for item in book.get_items():
            # Chapters are typically located in epub documents items.
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                all_chapter_texts = soup.findAll(text=True)
                chapter_text = u"\n".join(filter(self.check_tag, all_chapter_texts))
                text_list.append(chapter_text)

        text = "\n".join(text_list)
        return text
