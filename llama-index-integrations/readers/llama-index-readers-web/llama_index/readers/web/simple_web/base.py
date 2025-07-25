"""Simple Web scraper."""

from typing import List, Optional, Dict, Callable
import uuid
import requests

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class SimpleWebPageReader(BasePydanticReader):
    """
    Simple web page reader.

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
    _timeout: Optional[int] = PrivateAttr()
    _fail_on_error: bool = PrivateAttr()

    def __init__(
        self,
        html_to_text: bool = False,
        metadata_fn: Optional[Callable[[str], Dict]] = None,
        timeout: Optional[int] = 60,
        fail_on_error: bool = False,
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
        self._timeout = timeout
        self._fail_on_error = fail_on_error

    @classmethod
    def class_name(cls) -> str:
        return "SimpleWebPageReader"

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
        documents = []
        for url in urls:
            try:
                response = requests.get(url, headers=None, timeout=self._timeout)
            except Exception:
                if self._fail_on_error:
                    raise
                continue

            response_text = response.text

            if response.status_code != 200 and self._fail_on_error:
                raise ValueError(
                    f"Error fetching page from {url}. server returned status:"
                    f" {response.status_code} and response {response_text}"
                )

            if self.html_to_text:
                import html2text

                response_text = html2text.html2text(response_text)

            metadata: Dict = {"url": url}
            if self._metadata_fn is not None:
                metadata = self._metadata_fn(url)
                if "url" not in metadata:
                    metadata["url"] = url

            documents.append(
                Document(text=response_text, id_=str(uuid.uuid4()), metadata=metadata)
            )

        return documents
