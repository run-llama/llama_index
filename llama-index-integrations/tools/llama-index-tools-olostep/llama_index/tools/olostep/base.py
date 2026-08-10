"""Olostep tool spec."""

from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


def _unwrap(response):
    """Return the inner result object, whether the SDK nests it under
    `.result` or exposes fields directly on the response."""
    return getattr(response, "result", response)


class OlostepToolSpec(BaseToolSpec):
    """Olostep tool spec.

    Gives an agent live web access via Olostep: scrape any URL to clean
    markdown, get an AI-grounded answer with sources, crawl a website, and
    discover the URLs on a site. Requires an Olostep API key
    (https://www.olostep.com).
    """

    spec_functions = ["scrape", "answer", "crawl", "map_site"]

    def __init__(self, api_key: str) -> None:
        """Initialize with an Olostep API key."""
        from olostep import Olostep

        self.client = Olostep(api_key=api_key)

    def scrape(self, url: str) -> List[Document]:
        """
        Scrape a single URL and return its content as clean markdown.

        Args:
            url (str): The URL to scrape.

        Returns:
            List[Document]: A single Document with the page's markdown content
                and the source URL in metadata.

        """
        response = self.client.scrapes.create(
            url_to_scrape=url,
            formats=["markdown"],
        )
        result = _unwrap(response)
        content = getattr(result, "markdown_content", "") or ""
        return [Document(text=content, metadata={"url": url})]

    def answer(self, task: str) -> List[Document]:
        """
        Get an AI-grounded answer to a question or task, synthesized from live
        web sources.

        Args:
            task (str): The question or task to answer.

        Returns:
            List[Document]: A single Document containing the grounded answer.

        """
        response = self.client.answers.create(task=task)
        result = _unwrap(response)
        content = getattr(result, "json_content", None)
        if content is None:
            content = getattr(result, "markdown_content", None)
        if content is None:
            content = getattr(result, "answer", "") or ""
        return [Document(text=str(content), metadata={"task": task})]

    def crawl(self, start_url: str, max_pages: Optional[int] = 20) -> List[Document]:
        """
        Crawl a website starting from a URL and return the pages' content.

        Args:
            start_url (str): The URL to start crawling from.
            max_pages (Optional[int]): Maximum number of pages to crawl.
                Defaults to 20.

        Returns:
            List[Document]: One Document per crawled page, with the page URL in
                metadata.

        """
        crawl = self.client.crawls.create(
            start_url=start_url,
            max_pages=max_pages,
        )
        documents = []
        for page in crawl.pages():
            retrieved = self.client.retrieve(
                retrieve_id=page.retrieve_id,
                formats=["markdown"],
            )
            result = _unwrap(retrieved)
            content = getattr(result, "markdown_content", "") or ""
            page_url = getattr(page, "url", "") or ""
            documents.append(Document(text=content, metadata={"url": page_url}))
        return documents

    def map_site(self, url: str) -> List[Document]:
        """
        Discover all the URLs on a website.

        Args:
            url (str): The site URL to map.

        Returns:
            List[Document]: A single Document listing the discovered URLs, with
                the source site in metadata.

        """
        sitemap = self.client.maps.create(url=url)
        raw_urls = sitemap.urls
        # `urls` is a callable on the sitemap object.
        if callable(raw_urls):
            raw_urls = raw_urls()
        url_strings = []
        for u in raw_urls:
            if isinstance(u, str):
                url_strings.append(u)
            else:
                url_strings.append(getattr(u, "url", str(u)))
        return [Document(text="\n".join(url_strings), metadata={"url": url})]