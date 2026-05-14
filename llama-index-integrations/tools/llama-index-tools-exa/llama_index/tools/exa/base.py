"""Exa tool spec - web search API for AI."""

import datetime
from typing import List, Literal, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

SEARCH_TYPE = Literal["auto", "fast", "instant", "deep"]
CATEGORY = Literal[
    "company", "people", "research paper", "news", "personal site", "financial report"
]


class ExaToolSpec(BaseToolSpec):
    """Exa tool spec - web search API for AI."""

    spec_functions = [
        "search",
        "retrieve_documents",
        "search_and_retrieve_highlights",
        "search_and_retrieve_documents",
        "current_date",
    ]

    def __init__(
        self,
        api_key: str,
        verbose: bool = True,
        max_characters: int = 2000,
    ) -> None:
        """Initialize with parameters."""
        from exa_py import Exa

        self.client = Exa(api_key=api_key, user_agent="llama-index")
        self.client.headers["x-exa-integration"] = "llamaindex-integration"
        self._verbose = verbose
        self._max_characters = max_characters

    def search(
        self,
        query: str,
        num_results: Optional[int] = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        type: SEARCH_TYPE = "auto",
        category: Optional[CATEGORY] = None,
    ) -> List:
        """
        Exa allows you to use a natural language query to search the web.

        Args:
            query (str): A natural language query phrased as an answer for what the link provides, ie: "This is the latest news about space:"
            num_results (Optional[int]): Number of results to return. Defaults to 10.
            include_domains (Optional[List(str)]): A list of top level domains like ["wsj.com"] to limit the search to specific sites.
            exclude_domains (Optional[List(str)]): Top level domains to exclude.
            start_published_date (Optional[str]): A date string like "2020-06-15". Get the date from `current_date`
            end_published_date (Optional[str]): End date string
            type: Search type — auto, fast, instant, or deep.
            category: Optional category filter.

        """
        kwargs = {
            "num_results": num_results,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "start_published_date": start_published_date,
            "end_published_date": end_published_date,
            "type": type,
        }
        if category:
            kwargs["category"] = category

        response = self.client.search(query, **kwargs)
        return [
            {"title": result.title, "url": result.url, "id": result.id}
            for result in response.results
        ]

    def retrieve_documents(self, ids: List[str]) -> List[Document]:
        """
        Retrieve a list of document texts returned by `exa_search`, using the ID field.

        Args:
            ids (List(str)): the ids of the documents to retrieve

        """
        response = self.client.get_contents(ids)
        return [Document(text=result.text) for result in response.results]

    def find_similar(
        self,
        url: str,
        num_results: Optional[int] = 3,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
    ) -> List:
        """
        Retrieve a list of similar documents to a given url.

        Args:
            url (str): The web page to find similar results of
            num_results (Optional[int]): Number of results to return. Default 3.
            start_published_date (Optional[str]): A date string like "2020-06-15"
            end_published_date (Optional[str]): End date string

        """
        response = self.client.find_similar(
            url,
            num_results=num_results,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
        )
        return [
            {"title": result.title, "url": result.url, "id": result.id}
            for result in response.results
        ]

    def search_and_retrieve_documents(
        self,
        query: str,
        num_results: Optional[int] = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        type: SEARCH_TYPE = "auto",
        category: Optional[CATEGORY] = None,
    ) -> List[Document]:
        """
        Search and retrieve full-text documents.

        Args:
            query (str): the natural language query
            num_results (Optional[int]): Number of results. Defaults to 10.
            include_domains (Optional[List(str)]): A list of top level domains to search, like ["wsj.com"]
            exclude_domains (Optional[List(str)]): Top level domains to exclude.
            start_published_date (Optional[str]): A date string like "2020-06-15".
            end_published_date (Optional[str]): End date string
            type: Search type — auto, fast, instant, or deep.
            category: Optional category filter.

        """
        kwargs = {
            "num_results": num_results,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "start_published_date": start_published_date,
            "end_published_date": end_published_date,
            "text": {"max_characters": self._max_characters},
            "type": type,
        }
        if category:
            kwargs["category"] = category

        response = self.client.search_and_contents(query, **kwargs)
        return [Document(text=document.text) for document in response.results]

    def search_and_retrieve_highlights(
        self,
        query: str,
        num_results: Optional[int] = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        type: SEARCH_TYPE = "auto",
        category: Optional[CATEGORY] = None,
    ) -> List[Document]:
        """
        Search and retrieve highlights (query-relevant excerpts).

        Args:
            query (str): the natural language query
            num_results (Optional[int]): Number of results. Defaults to 10.
            include_domains (Optional[List(str)]): A list of top level domains to search, like ["wsj.com"]
            exclude_domains (Optional[List(str)]): Top level domains to exclude.
            start_published_date (Optional[str]): A date string like "2020-06-15".
            end_published_date (Optional[str]): End date string
            type: Search type — auto, fast, instant, or deep.
            category: Optional category filter.

        """
        kwargs = {
            "num_results": num_results,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "start_published_date": start_published_date,
            "end_published_date": end_published_date,
            "highlights": True,
            "type": type,
        }
        if category:
            kwargs["category"] = category

        response = self.client.search_and_contents(query, **kwargs)
        return [Document(text=document.highlights[0]) for document in response.results]

    def current_date(self):
        """
        A function to return todays date.

        Call this before any other functions that take timestamps as an argument
        """
        return datetime.date.today()
