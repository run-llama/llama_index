"""Simple reader that reads wikipedia."""
from typing import Any, List

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class WikipediaReader(BaseReader):
    """Wikipedia reader.

    Reads a page.

    """

    def load_data(
        self, pages: List[str], lang: str = "en", **load_kwargs: Any
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            pages (List[str]): List of pages to read.
            lang  (str): language of wikipedia texts (default English)
        """
        import wikipedia

        results = []
        for page in pages:
            wikipedia.set_lang(lang)
            page_content = wikipedia.page(page, **load_kwargs).content
            results.append(Document(text=page_content))
        return results
