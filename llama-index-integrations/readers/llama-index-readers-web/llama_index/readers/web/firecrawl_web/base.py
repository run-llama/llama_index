"""Firecrawl Web Reader."""

from typing import Any, List, Optional, Dict, Callable, Union, Literal
from pydantic import Field

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class FireCrawlWebReader(BasePydanticReader):
    """
    FireCrawl Web Reader for converting URLs to LLM-accessible markdown using Firecrawl.dev.

    This reader supports four modes:
    - "scrape": Extract content from a single URL
    - "crawl": Crawl an entire website and extract content from all accessible pages
    - "search": Search for content across the web
    - "extract": Extract structured data from URLs using a prompt

    Args:
        api_key (str): The Firecrawl API key required for authentication.
        api_url (Optional[str]): Custom API URL for local deployment. Defaults to None.
        mode (str): The operation mode. Options: "scrape", "crawl", "search", "extract". Defaults to "crawl".
        params (Optional[Dict]): Parameters to pass to the Firecrawl API methods.

    Example:
        ```python
        from llama_index.readers.web import FireCrawlWebReader

        # Scrape a single page
        reader = FireCrawlWebReader(
            api_key="your_api_key",
            mode="scrape",
            params={"timeout": 30}
        )
        documents = reader.load_data(url="https://example.com")

        # Crawl a website
        reader = FireCrawlWebReader(
            api_key="your_api_key",
            mode="crawl",
            params={"max_depth": 2, "limit": 10}
        )
        documents = reader.load_data(url="https://example.com")

        # Search for content
        reader = FireCrawlWebReader(
            api_key="your_api_key",
            mode="search",
            params={"limit": 5}
        )
        documents = reader.load_data(query="Who is the president of the United States?")

        # Extract structured data
        reader = FireCrawlWebReader(
            api_key="your_api_key",
            mode="extract",
            params={"prompt": "Extract the main topics"}
        )
        documents = reader.load_data(urls=["https://example.com"])
        ```
    """

    api_key: str = Field(description="The Firecrawl API key")
    api_url: Optional[str] = Field(None, description="Custom API URL for local deployment")
    mode: str = Field("crawl", description="Operation mode: scrape, crawl, search, or extract")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters to pass to the Firecrawl API")

    _firecrawl: Any = PrivateAttr()
    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        mode: str = "crawl",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the FireCrawlWebReader with the specified configuration."""
        # Validate mode
        valid_modes = ["scrape", "crawl", "search", "extract"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # Initialize the parent class
        super().__init__(
            api_key=api_key,
            api_url=api_url,
            mode=mode,
            params=params,
        )

        # Import and initialize FirecrawlApp
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )

        # Initialize FirecrawlApp
        if api_url:
            self._firecrawl = FirecrawlApp(api_key=api_key, api_url=api_url)
        else:
            self._firecrawl = FirecrawlApp(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "FireCrawlWebReader"

    def load_data(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        urls: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load data from Firecrawl based on the specified mode.

        Args:
            url (Optional[str]): URL to scrape or crawl. Required for scrape and crawl modes.
            query (Optional[str]): Query to search for. Required for search mode.
            urls (Optional[List[str]]): List of URLs for extract mode. Required for extract mode.

        Returns:
            List[Document]: List of documents with extracted content and metadata.

        Raises:
            ValueError: If invalid combination of parameters is provided or required parameters are missing.
        """
        # Validate input parameters
        if self.mode in ["scrape", "crawl"] and url is None:
            raise ValueError(f"URL must be provided for {self.mode} mode.")
        elif self.mode == "search" and query is None:
            raise ValueError("Query must be provided for search mode.")
        elif self.mode == "extract" and urls is None:
            raise ValueError("URLs must be provided for extract mode.")

        documents = []

        if self.mode == "scrape":
            documents = self._scrape_url(url)
        elif self.mode == "crawl":
            documents = self._crawl_url(url)
        elif self.mode == "search":
            documents = self._search_content(query)
        elif self.mode == "extract":
            documents = self._extract_data(urls)

        return documents

    def _scrape_url(self, url: str) -> List[Document]:
        """Scrape content from a single URL."""
        try:
            # Extract scrape options and pass as keyword arguments
            scrape_kwargs = self.params or {}
            scrape_kwargs["integration"] = "llamaindex"
            
            # Call the scrape method with keyword arguments
            response = self._firecrawl.scrape_url(url, **scrape_kwargs)
            
            # Handle response
            if isinstance(response, dict):
                return [
                    Document(
                        text=response.get("markdown", ""),
                        metadata={
                            "url": url,
                            "source": "scrape",
                            **response.get("metadata", {})
                        }
                    )
                ]
            else:
                # Handle object response
                return [
                    Document(
                        text=getattr(response, "markdown", ""),
                        metadata={
                            "url": url,
                            "source": "scrape",
                            **getattr(response, "metadata", {})
                        }
                    )
                ]
        except Exception as e:
            return [
                Document(
                    text=f"Error scraping {url}: {str(e)}",
                    metadata={"url": url, "source": "scrape", "error": str(e)}
                )
            ]

    def _crawl_url(self, url: str) -> List[Document]:
        """Crawl an entire website starting from the given URL."""
        try:
            # Extract crawl options and pass as keyword arguments
            crawl_kwargs = self.params or {}
            crawl_kwargs["integration"] = "llamaindex"
            
            # Call the crawl method with keyword arguments
            response = self._firecrawl.crawl_url(url, **crawl_kwargs)
            
            documents = []
            
            # Handle response
            if isinstance(response, dict):
                data = response.get("data", [])
                for doc in data:
                    documents.append(
                        Document(
                            text=doc.get("markdown", ""),
                            metadata={
                                "url": doc.get("url", url),
                                "source": "crawl",
                                **doc.get("metadata", {})
                            }
                        )
                    )
            else:
                # Handle object response - FirecrawlDocument objects
                data = getattr(response, "data", [])
                for doc in data:
                    documents.append(
                        Document(
                            text=getattr(doc, "markdown", ""),
                            metadata={
                                "url": getattr(doc, "url", url),
                                "source": "crawl",
                                **getattr(doc, "metadata", {})
                            }
                        )
                    )
            
            return documents
        except Exception as e:
            return [
                Document(
                    text=f"Error crawling {url}: {str(e)}",
                    metadata={"url": url, "source": "crawl", "error": str(e)}
                )
            ]

    def _search_content(self, query: str) -> List[Document]:
        """Search for content across the web."""
        try:
            # Extract search options and pass as keyword arguments
            search_kwargs = self.params or {}
            search_kwargs["integration"] = "llamaindex"
            
            # Call the search method with keyword arguments
            response = self._firecrawl.search(query, **search_kwargs)
            
            documents = []
            
            # Handle response
            if isinstance(response, dict):
                if response.get("success", False):
                    data = response.get("data", [])
                    for result in data:
                        # Search results have title, description, url but no markdown
                        text = result.get("description", "")
                        documents.append(
                            Document(
                                text=text,
                                metadata={
                                    "title": result.get("title", ""),
                                    "url": result.get("url", ""),
                                    "description": result.get("description", ""),
                                    "source": "search",
                                    "query": query,
                                    **result.get("metadata", {})
                                }
                            )
                        )
                else:
                    warning = response.get("warning", "Unknown error")
                    documents.append(
                        Document(
                            text=f"Search for '{query}' was unsuccessful: {warning}",
                            metadata={"source": "search", "query": query, "error": warning}
                        )
                    )
            else:
                # Handle object response
                if hasattr(response, "success") and response.success:
                    data = getattr(response, "data", [])
                    for result in data:
                        # Search results have title, description, url but no markdown
                        text = getattr(result, "description", "")
                        documents.append(
                            Document(
                                text=text,
                                metadata={
                                    "title": result.get("title") or "",
                                    "url": result.get("url") or "",
                                    "description": result.get("description") or "",
                                    "source": "search",
                                    "query": query
                                }
                            )
                        )
            
            return documents
        except Exception as e:
            return [
                Document(
                    text=f"Error searching for '{query}': {str(e)}",
                    metadata={"source": "search", "query": query, "error": str(e)}
                )
            ]

    def _extract_data(self, urls: List[str]) -> List[Document]:
        """Extract structured data from URLs using a prompt."""
        try:
            # Extract extract options and pass as keyword arguments
            extract_kwargs = self.params or {}
            extract_kwargs["integration"] = "llamaindex"
            
            # Ensure prompt is provided
            if "prompt" not in extract_kwargs:
                raise ValueError("A 'prompt' parameter is required for extract mode.")
            
            # Call the extract method with keyword arguments
            response = self._firecrawl.extract(urls, **extract_kwargs)
            
            documents = []
            
            # Handle response
            if isinstance(response, dict):
                if response.get("success", False):
                    data = response.get("data", {})
                    sources = response.get("sources", {})
                    
                    if data:
                        # Convert extracted data to text
                        text_parts = []
                        for key, value in data.items():
                            text_parts.append(f"{key}: {value}")
                        
                        text = "\n".join(text_parts)
                        
                        documents.append(
                            Document(
                                text=text,
                                metadata={
                                    "urls": urls,
                                    "source": "extract",
                                    "status": response.get("status"),
                                    "expires_at": response.get("expiresAt"),
                                    "sources": sources
                                }
                            )
                        )
                    else:
                        documents.append(
                            Document(
                                text="Extraction was successful but no data was returned",
                                metadata={"urls": urls, "source": "extract"}
                            )
                        )
                else:
                    warning = response.get("warning", "Unknown error")
                    documents.append(
                        Document(
                            text=f"Extraction was unsuccessful: {warning}",
                            metadata={"urls": urls, "source": "extract", "error": warning}
                        )
                    )
            else:
                # Handle object response
                if hasattr(response, "success") and response.success:
                    data = getattr(response, "data", {})
                    if data:
                        text_parts = []
                        for key, value in data.items():
                            text_parts.append(f"{key}: {value}")
                        
                        text = "\n".join(text_parts)
                        
                        documents.append(
                            Document(
                                text=text,
                                metadata={
                                    "urls": urls,
                                    "source": "extract"
                                }
                            )
                        )
            
            return documents
        except Exception as e:
            return [
                Document(
                    text=f"Error extracting data from {urls}: {str(e)}",
                    metadata={"urls": urls, "source": "extract", "error": str(e)}
                )
            ]
