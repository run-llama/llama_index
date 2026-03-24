"""Olostep tool spec."""

from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class OlostepToolSpec(BaseToolSpec):
    """Olostep tool spec for web scraping, crawling, and search."""

    spec_functions = [
        "scrape_url",
        "crawl_website",
        "map_website",
        "search_web",
        "answer_question",
        "batch_scrape",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize with Olostep API key.

        Args:
            api_key: Olostep API key. If not provided, falls back to OLOSTEP_API_KEY env var.

        """
        from olostep import Olostep

        self.api_key = api_key
        self.client = Olostep(api_key=api_key)

    def scrape_url(
        self,
        url: str,
        formats: str = "markdown",
        wait_before_scraping: int = 0,
        country: Optional[str] = None,
        parser_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Scrape a single URL and return its content as a Document.

        Use this to extract the content of any webpage. Returns clean
        markdown by default. Supports JS-rendered sites.

        Args:
            url: The full URL to scrape.
            formats: Comma-separated formats: "markdown" (default), "html", "text", "json".
            wait_before_scraping: Milliseconds to wait for JS rendering.
            country: Two-letter country code for geo-location (e.g. "us", "gb").
            parser_id: Pre-built parser for structured JSON extraction.
                       Options: "@olostep/google-search", "@olostep/amazon-it-product",
                       "@olostep/extract-emails", "@olostep/extract-socials".

        Returns:
            List with a single Document containing the page content.

        """
        try:
            from olostep import Olostep_BaseError

            # Parse formats into list
            fmt_list = [fmt.strip() for fmt in formats.split(",")]

            # Build kwargs
            kwargs = {
                "url_to_scrape": url,
                "formats": fmt_list,
            }
            if wait_before_scraping > 0:
                kwargs["wait_before_scraping"] = wait_before_scraping
            if country:
                kwargs["country"] = country
            if parser_id:
                kwargs["parser_id"] = parser_id

            # Call API
            response = self.client.scrapes.create(**kwargs)

            # Extract content based on first format
            content = ""
            if fmt_list:
                format_key = fmt_list[0].lower()
                if format_key == "markdown":
                    content = response.markdown or ""
                elif format_key == "html":
                    content = response.html or ""
                elif format_key == "text":
                    content = response.text or ""
                elif format_key == "json":
                    content = str(response.json) or ""

            return [
                Document(
                    text=content,
                    extra_info={"url": url, "format": formats},
                )
            ]

        except Exception as e:
            from olostep import Olostep_BaseError

            if isinstance(e, Olostep_BaseError):
                return [
                    Document(
                        text=f"Error scraping URL: {e!s}",
                        extra_info={"url": url, "error": str(e)},
                    )
                ]
            raise

    def crawl_website(
        self,
        url: str,
        max_pages: int = 20,
        include_urls: Optional[str] = None,
        exclude_urls: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[Document]:
        """
        Crawl a website and return content from all crawled pages as Documents.

        Use this to gather content from an entire website or a section of it.
        Each crawled page becomes a separate Document. Best for building
        knowledge bases from documentation sites or blogs.

        Args:
            url: The starting URL for the crawl.
            max_pages: Maximum number of pages to crawl (default: 20, max: 1000).
            include_urls: Comma-separated glob patterns to include (e.g. "/blog/**, /docs/**").
            exclude_urls: Comma-separated glob patterns to exclude (e.g. "/admin/**").
            search_query: Filter crawled pages by relevance to this query.

        Returns:
            List of Documents, one per crawled page.

        """
        try:
            from olostep import Olostep_BaseError

            # Build kwargs
            kwargs = {
                "start_url": url,
                "max_pages": max_pages,
            }
            if include_urls:
                kwargs["include_urls"] = include_urls
            if exclude_urls:
                kwargs["exclude_urls"] = exclude_urls
            if search_query:
                kwargs["search_query"] = search_query

            # Call API
            crawl = self.client.crawls.create(**kwargs)

            # Collect pages
            documents = []
            for page in crawl.pages():
                markdown_content = page.markdown or ""
                documents.append(
                    Document(
                        text=markdown_content,
                        extra_info={"url": page.url},
                    )
                )

            return documents

        except Exception as e:
            from olostep import Olostep_BaseError

            if isinstance(e, Olostep_BaseError):
                return [
                    Document(
                        text=f"Error crawling website: {e!s}",
                        extra_info={"url": url, "error": str(e)},
                    )
                ]
            raise

    def map_website(
        self,
        url: str,
        include_urls: Optional[str] = None,
        exclude_urls: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[Document]:
        """
        Discover all URLs on a website and return them as a Document.

        Use this to explore a site's structure before deciding which pages
        to scrape. Returns URLs from sitemaps and discovered links.

        Args:
            url: The website URL to map.
            include_urls: Comma-separated glob patterns to include.
            exclude_urls: Comma-separated glob patterns to exclude.
            top_n: Limit number of URLs returned.

        Returns:
            List with a single Document containing all discovered URLs.

        """
        try:
            from olostep import Olostep_BaseError

            # Build kwargs
            kwargs = {"url": url}
            if include_urls:
                kwargs["include_urls"] = include_urls
            if exclude_urls:
                kwargs["exclude_urls"] = exclude_urls

            # Call API
            map_result = self.client.maps.create(**kwargs)

            # Collect URLs
            urls = []
            for url_item in map_result.urls():
                urls.append(url_item.url)

            # Limit to top_n if specified
            if top_n:
                urls = urls[:top_n]

            urls_text = "\n".join(urls)

            return [
                Document(
                    text=urls_text,
                    extra_info={"url": url, "total_urls": len(urls)},
                )
            ]

        except Exception as e:
            from olostep import Olostep_BaseError

            if isinstance(e, Olostep_BaseError):
                return [
                    Document(
                        text=f"Error mapping website: {e!s}",
                        extra_info={"url": url, "error": str(e)},
                    )
                ]
            raise

    def search_web(self, query: str) -> List[Document]:
        """
        Search the web and return relevant links as Documents.

        Use this to find web pages on a topic. Unlike answer_question
        (which synthesizes an answer), this returns raw links for further
        investigation. Each result becomes a Document.

        Args:
            query: Natural language search query.

        Returns:
            List of Documents, one per search result, each containing
            the page title and description with the URL in extra_info.

        """
        try:
            from olostep import Olostep_BaseError

            # Call API
            result = self.client.searches.create(query=query)

            # Collect results
            documents = []
            if result.result and result.result.links:
                for link in result.result.links:
                    text = (
                        f"{link.title}\n{link.description}"
                        if link.description
                        else link.title
                    )
                    documents.append(
                        Document(
                            text=text,
                            extra_info={"url": link.url},
                        )
                    )

            return documents

        except Exception as e:
            from olostep import Olostep_BaseError

            if isinstance(e, Olostep_BaseError):
                return [
                    Document(
                        text=f"Error searching web: {e!s}",
                        extra_info={"query": query, "error": str(e)},
                    )
                ]
            raise

    def answer_question(
        self,
        task: str,
        json_schema: Optional[str] = None,
    ) -> List[Document]:
        """
        Search the web and return an AI-synthesized answer as a Document.

        Use this for research tasks and fact-checking where you need a
        synthesized answer with sources rather than raw links. Olostep
        searches the web, validates sources, and returns a grounded answer.

        Args:
            task: The question or research task in natural language.
            json_schema: Optional JSON string schema for structured output
                         (e.g. '{"ceo": "", "founded": "", "valuation": ""}').
                         Returns NOT_FOUND for unverifiable fields.

        Returns:
            List with a single Document containing the answer and sources.

        """
        try:
            from olostep import Olostep_BaseError

            # Build kwargs
            kwargs = {"task": task}
            if json_schema:
                kwargs["json_schema"] = json_schema

            # Call API
            answer = self.client.answers.create(**kwargs)

            # Extract answer text and sources
            answer_text = answer.answer or ""
            sources = []
            if answer.sources:
                sources = [source.url for source in answer.sources]

            return [
                Document(
                    text=answer_text,
                    extra_info={"sources": sources, "task": task},
                )
            ]

        except Exception as e:
            from olostep import Olostep_BaseError

            if isinstance(e, Olostep_BaseError):
                return [
                    Document(
                        text=f"Error answering question: {e!s}",
                        extra_info={"task": task, "error": str(e)},
                    )
                ]
            raise

    def batch_scrape(
        self,
        urls: str,
        formats: str = "markdown",
        parser_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Scrape multiple URLs concurrently and return each as a Document.

        Use this when you have many URLs to scrape at once (50-10,000).
        Processing takes ~5-8 minutes regardless of batch size, making it
        far more efficient than individual scrapes for large URL lists.

        Args:
            urls: Comma-separated list of URLs to scrape.
            formats: Output formats: "markdown", "html", "text", "json".
            parser_id: Parser ID for structured extraction from each URL.

        Returns:
            List of Documents, one per URL scraped.

        """
        try:
            from olostep import Olostep_BaseError

            # Parse URLs and formats
            url_list = [u.strip() for u in urls.split(",")]
            fmt_list = [fmt.strip() for fmt in formats.split(",")]

            # Build kwargs
            kwargs = {
                "urls": url_list,
                "formats": fmt_list,
            }
            if parser_id:
                kwargs["parser_id"] = parser_id

            # Call API
            batch = self.client.batches.create(**kwargs)

            # Collect results
            documents = []
            for item in batch.items():
                content = ""
                if fmt_list:
                    format_key = fmt_list[0].lower()
                    if format_key == "markdown":
                        content = item.markdown or ""
                    elif format_key == "html":
                        content = item.html or ""
                    elif format_key == "text":
                        content = item.text or ""
                    elif format_key == "json":
                        content = str(item.json) or ""

                documents.append(
                    Document(
                        text=content,
                        extra_info={"url": item.url, "format": formats},
                    )
                )

            return documents

        except Exception as e:
            from olostep import Olostep_BaseError

            if isinstance(e, Olostep_BaseError):
                return [
                    Document(
                        text=f"Error batch scraping: {e!s}",
                        extra_info={"urls": urls, "error": str(e)},
                    )
                ]
            raise
