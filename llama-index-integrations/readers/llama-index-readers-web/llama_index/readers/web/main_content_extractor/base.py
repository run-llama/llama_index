from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MainContentExtractorReader(BaseReader):
    """
    MainContentExtractor web page reader.

    Reads pages from the web.

    Args:
        text_format (str, optional): The format of the text. Defaults to "markdown".
            Requires `MainContentExtractor` package.

    """

    def __init__(self, text_format: str = "markdown") -> None:
        """Initialize with parameters."""
        self.text_format = text_format

    def load_data(self, urls: List[str]) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")

        from main_content_extractor import MainContentExtractor

        documents = []
        for url in urls:
            response = requests.get(url).text
            response = MainContentExtractor.extract(
                response, output_format=self.text_format, include_links=False
            )

            documents.append(Document(text=response))

        return documents
