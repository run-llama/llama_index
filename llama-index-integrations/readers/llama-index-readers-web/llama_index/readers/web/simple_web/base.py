"""Simple Web scraper."""
from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class SimpleWebPageReader(BaseReader):
    """Simple web page reader.

    Reads pages from the web.

    Args:
        html_to_text (bool): Whether to convert HTML to text.
            Requires `html2text` package.

    """

    def __init__(self, html_to_text: bool = False) -> None:
        """Initialize with parameters."""
        self._html_to_text = html_to_text

    def load_data(self, urls: List[str]) -> List[Document]:
        """Load data from the input directory.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")

        documents = []
        for url in urls:
            response = requests.get(url).text
            if self._html_to_text:
                import html2text

                response = html2text.html2text(response)

            documents.append(Document(text=response))

        return documents
