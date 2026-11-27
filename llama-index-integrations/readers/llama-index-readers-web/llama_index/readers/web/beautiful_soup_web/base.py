"""Beautiful Soup Web scraper."""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


def _substack_reader(soup: Any, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """Extract text from Substack blog post."""
    extra_info = {}
    title = soup.select_one("h1.post-title")
    if title is not None:
        extra_info["Title of this Substack post"] = title.getText()
    subtitle = soup.select_one("h3.subtitle")
    if subtitle is not None:
        extra_info["Subtitle"] = subtitle.getText()
    author = soup.select_one("span.byline-names")
    if author is not None:
        extra_info["Author"] = author.getText()
    content = soup.select_one("div.available-content")
    text = content.getText() if content is not None else soup.getText()
    return text, extra_info


def _readthedocs_reader(soup: Any, url: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """Extract text from a ReadTheDocs documentation site."""
    import requests
    from bs4 import BeautifulSoup

    links = soup.find_all("a", {"class": "reference internal"})
    rtd_links = []

    for link in links:
        href = link.get("href")
        if href is not None:
            rtd_links.append(href)
    for i in range(len(rtd_links)):
        if not rtd_links[i].startswith("http"):
            rtd_links[i] = urljoin(url, rtd_links[i])

    texts = []
    for doc_link in rtd_links:
        page_link = requests.get(doc_link)
        soup = BeautifulSoup(page_link.text, "html.parser")
        main = soup.find(attrs={"role": "main"})
        text = main.get_text() if main is not None else None
        if text:
            texts.append("\n".join([t for t in text.split("\n") if t]))
    return "\n".join(texts), {}


def _readmedocs_reader(
    soup: Any, url: str, include_url_in_text: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """Extract text from a ReadMe documentation site."""
    import requests
    from bs4 import BeautifulSoup

    links = soup.find_all("a")
    docs_links = [
        link["href"] for link in links if link.get("href") and "/docs/" in link["href"]
    ]
    docs_links = list(set(docs_links))
    for i in range(len(docs_links)):
        if not docs_links[i].startswith("http"):
            docs_links[i] = urljoin(url, docs_links[i])

    texts = []
    for doc_link in docs_links:
        page_link = requests.get(doc_link)
        soup = BeautifulSoup(page_link.text, "html.parser")
        try:
            text = ""
            for element in soup.find_all("article", {"id": "content"}):
                for child in element.descendants:
                    if child.name == "a" and child.has_attr("href"):
                        if include_url_in_text:
                            url = child.get("href")
                            if url is not None and "edit" in url:
                                text += child.text
                            else:
                                text += (
                                    f"{child.text} (Reference url: {doc_link}{url}) "
                                )
                    elif child.string and child.string.strip():
                        text += child.string.strip() + " "

        except IndexError:
            text = None
            logger.error(f"Could not extract text from {doc_link}")
            continue
        texts.append("\n".join([t for t in text.split("\n") if t]))
    return "\n".join(texts), {}


def _gitbook_reader(
    soup: Any, url: str, include_url_in_text: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """Extract text from a ReadMe documentation site."""
    import requests
    from bs4 import BeautifulSoup

    links = soup.find_all("a")
    docs_links = [
        link["href"] for link in links if link.get("href") and "/docs/" in link["href"]
    ]
    docs_links = list(set(docs_links))
    for i in range(len(docs_links)):
        if not docs_links[i].startswith("http"):
            docs_links[i] = urljoin(url, docs_links[i])

    texts = []
    for doc_link in docs_links:
        page_link = requests.get(doc_link)
        soup = BeautifulSoup(page_link.text, "html.parser")
        main = soup.find("main")
        if main is None:
            logger.error(f"Could not extract text from {doc_link}")
            continue
        clean_text = ", ".join([tag.get_text() for tag in main])
        texts.append(clean_text)
    return "\n".join(texts), {}


DEFAULT_WEBSITE_EXTRACTOR: Dict[
    str, Callable[[Any, str], Tuple[str, Dict[str, Any]]]
] = {
    "substack.com": _substack_reader,
    "readthedocs.io": _readthedocs_reader,
    "readme.com": _readmedocs_reader,
    "gitbook.io": _gitbook_reader,
}


class BeautifulSoupWebReader(BasePydanticReader):
    """
    BeautifulSoup web page reader.

    Reads pages from the web.
    Requires the `bs4` and `urllib` packages.

    Args:
        website_extractor (Optional[Dict[str, Callable]]): A mapping of website
            hostname (e.g. google.com) to a function that specifies how to
            extract text from the BeautifulSoup obj. See DEFAULT_WEBSITE_EXTRACTOR.

    """

    is_remote: bool = True
    _website_extractor: Dict[str, Callable] = PrivateAttr()

    def __init__(self, website_extractor: Optional[Dict[str, Callable]] = None) -> None:
        super().__init__()
        self._website_extractor = website_extractor or DEFAULT_WEBSITE_EXTRACTOR

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "BeautifulSoupWebReader"

    def load_data(
        self,
        urls: List[str],
        custom_hostname: Optional[str] = None,
        include_url_in_text: Optional[bool] = True,
    ) -> List[Document]:
        """
        Load data from the urls.

        Args:
            urls (List[str]): List of URLs to scrape.
            custom_hostname (Optional[str]): Force a certain hostname in the case
                a website is displayed under custom URLs (e.g. Substack blogs)
            include_url_in_text (Optional[bool]): Include the reference url in the text of the document

        Returns:
            List[Document]: List of documents.

        """
        from urllib.parse import urlparse

        import requests
        from bs4 import BeautifulSoup

        documents = []
        for url in urls:
            try:
                page = requests.get(url)
            except Exception:
                raise ValueError(f"One of the inputs is not a valid url: {url}")

            hostname = custom_hostname or urlparse(url).hostname or ""

            soup = BeautifulSoup(page.content, "html.parser")

            data = ""
            extra_info = {"URL": url}
            if hostname in self._website_extractor:
                data, metadata = self._website_extractor[hostname](
                    soup=soup, url=url, include_url_in_text=include_url_in_text
                )
                extra_info.update(metadata)

            else:
                data = soup.getText()

            documents.append(
                Document(text=data, id_=str(uuid.uuid4()), extra_info=extra_info)
            )

        return documents
