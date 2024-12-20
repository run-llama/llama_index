"""Simple Web scraper."""
from typing import List, Optional, Dict, Callable

import requests

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class SimpleWebPageReader(BasePydanticReader):
    """Simple web page reader.

    Reads pages from the web.

    Args:
        html_to_text (bool): Whether to convert HTML to text.
            Requires `html2text` package.
        metadata_fn (Optional[Callable[[str], Dict]]): A function that takes in
            a URL and returns a dictionary of metadata.
            Default is None.
    """

    is_remote: bool = True
    html_to_text: bool

    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    def __init__(
        self,
        html_to_text: bool = False,
        metadata_fn: Optional[Callable[[str], Dict]] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            import html2text  # noqa
        except ImportError:
            raise ImportError(
                "`html2text` package not found, please run `pip install html2text`"
            )
        super().__init__(html_to_text=html_to_text)
        self._metadata_fn = metadata_fn

    @classmethod
    def class_name(cls) -> str:
        return "SimpleWebPageReader"

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
            response = requests.get(url, headers=None).text
            if self.html_to_text:
                import html2text

                response = html2text.html2text(response)

            metadata: Optional[Dict] = None
            if self._metadata_fn is not None:
                metadata = self._metadata_fn(url)

            documents.append(Document(text=response, id_=url, metadata=metadata or {}))

        return documents
