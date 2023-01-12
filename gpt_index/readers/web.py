"""Web scraper."""
import logging
from typing import List

from langchain.utilities import RequestsWrapper

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document

logger = logging.getLogger(__name__)


class SimpleWebPageReader(BaseReader):
    """Simple web page reader.

    Reads pages from the web.

    Args:
        html_to_text (bool): Whether to convert HTML to text.
            Requires `html2text` package.

    """

    def __init__(self, html_to_text: bool = False) -> None:
        """Initialize with parameters."""
        try:
            import html2text  # noqa: F401
        except ImportError:
            raise ValueError(
                "`html2text` package not found, please run `pip install html2text`"
            )
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
        requests = RequestsWrapper()
        documents = []
        for url in urls:
            response = requests.run(url)
            if self._html_to_text:
                import html2text

                response = html2text.html2text(response)

            documents.append(Document(response))

        return documents


class TrafilaturaWebReader(BaseReader):
    """Trafilatura web page reader.

    Reads pages from the web.
    Requires the `trafilatura` package.

    """

    def __init__(self) -> None:
        """Initialize with parameters."""
        try:
            import trafilatura  # noqa: F401
        except ImportError:
            raise ValueError(
                "`trafilatura` package not found, please run `pip install trafilatura`"
            )

    def load_data(self, urls: List[str]) -> List[Document]:
        """Load data from the input directory.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        import trafilatura

        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        documents = []
        for url in urls:
            downloaded = trafilatura.fetch_url(url)
            response = trafilatura.extract(downloaded)
            print(response)
            documents.append(Document(response))

        return documents


if __name__ == "__main__":
    reader = SimpleWebPageReader()
    print(reader.load_data(["http://www.google.com"]))
