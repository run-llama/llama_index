from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

if TYPE_CHECKING:
    from bs4 import Tag


class HTMLTagReader(BaseReader):
    """
    Read HTML files and extract text from a specific tag with BeautifulSoup.

    By default, reads the text from the ``<section>`` tag.
    """

    def __init__(
        self,
        tag: str = "section",
        ignore_no_id: bool = False,
    ) -> None:
        self._tag = tag
        self._ignore_no_id = ignore_no_id

        super().__init__()

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("bs4 is required to read HTML files.")

        with open(file, encoding="utf-8") as html_file:
            soup = BeautifulSoup(html_file, "html.parser")

        tags = soup.find_all(self._tag)
        docs = []
        for tag in tags:
            tag_id = tag.get("id")
            tag_text = self._extract_text_from_tag(tag)

            if self._ignore_no_id and not tag_id:
                continue

            metadata = {
                "tag": self._tag,
                "tag_id": tag_id,
                "file_path": str(file),
            }
            metadata.update(extra_info or {})

            doc = Document(
                text=tag_text,
                metadata=metadata,
            )
            docs.append(doc)
        return docs

    def _extract_text_from_tag(self, tag: "Tag") -> str:
        try:
            from bs4 import NavigableString
        except ImportError:
            raise ImportError("bs4 is required to read HTML files.")

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
