import httpx
from typing import List

from defusedxml.ElementTree import fromstring
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.web.async_web.base import AsyncWebPageReader


class SitemapReader(BaseReader):
    """
    Asynchronous sitemap reader for web.

    Reads pages from the web based on their sitemap.xml.

    Args:
        sitemap_url (string): Path to the sitemap.xml. e.g. https://gpt-index.readthedocs.io/sitemap.xml
        html_to_text (bool): Whether to convert HTML to text.
            Requires `html2text` package.
        limit (int): Maximum number of concurrent requests.

    """

    xml_schema_sitemap = "http://www.sitemaps.org/schemas/sitemap/0.9"

    def __init__(self, html_to_text: bool = False, limit: int = 10) -> None:
        """Initialize with parameters."""
        self._async_loader = AsyncWebPageReader(html_to_text=html_to_text, limit=limit)
        self._html_to_text = html_to_text
        self._limit = limit

    def _load_sitemap(self, sitemap_url: str) -> str:
        sitemap_url_request = httpx.get(sitemap_url)

        return sitemap_url_request.content

    def _parse_sitemap(self, raw_sitemap: str, filter_locs: str = None) -> list:
        sitemap = fromstring(raw_sitemap)
        sitemap_urls = []

        for url in sitemap.findall(f"{{{self.xml_schema_sitemap}}}url"):
            location = url.find(f"{{{self.xml_schema_sitemap}}}loc").text

            if filter_locs is None or filter_locs in location:
                sitemap_urls.append(location)

        return sitemap_urls

    def load_data(self, sitemap_url: str, filter: str = None) -> List[Document]:
        sitemap = self._load_sitemap(sitemap_url=sitemap_url)
        sitemap_urls = self._parse_sitemap(sitemap, filter)

        return self._async_loader.load_data(urls=sitemap_urls)
