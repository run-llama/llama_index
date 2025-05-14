"""Metaphor tool spec."""

import datetime
from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class MetaphorToolSpec(BaseToolSpec):
    """Metaphor tool spec."""

    spec_functions = [
        "search",
        "retrieve_documents",
        "search_and_retrieve_documents",
        "find_similar",
        "current_date",
    ]

    def __init__(self, api_key: str, verbose: bool = True) -> None:
        """Initialize with parameters."""
        from metaphor_python import Metaphor

        self.client = Metaphor(api_key=api_key, user_agent="llama-index")
        self._verbose = verbose

    def search(
        self,
        query: str,
        num_results: Optional[int] = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
    ) -> str:
        """
        Metaphor allows you to use a natural language query to search the internet.

        Args:
            query (str): A natural language query phrased as an answer for what the link provides, ie: "This is the latest news about space:"
            num_results (Optional[int]): Number of results to return. Defaults to 10.
            include_domains (Optional[List(str)]): A list of top level domains like ["wsj.com"] to limit the search to specific sites.
            exclude_domains (Optional[List(str)]): Top level domains to exclude.
            start_published_date (Optional[str]): A date string like "2020-06-15". Get the date from `current_date`
            end_published_date (Optional[str]): End date string

        """
        response = self.client.search(
            query,
            num_results=num_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
            use_autoprompt=True,
        )
        if self._verbose:
            print(f"[Metaphor Tool] Autoprompt: {response.autoprompt_string}")
        return [
            {"title": result.title, "url": result.url, "id": result.id}
            for result in response.results
        ]

    def retrieve_documents(self, ids: List[str]) -> List[Document]:
        """
        Retrieve a list of document summaries returned by `metaphor_search`, using the ID field.

        Args:
            ids (List(str)): the ids of the documents to retrieve

        """
        response = self.client.get_contents(ids)
        return [Document(text=con.extract) for con in response.contents]

    def find_similar(
        self,
        url: str,
        num_results: Optional[int] = 3,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
    ) -> str:
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
    ) -> str:
        """
        Combines the functionality of `search` and `retrieve_documents`.

        Args:
            query (str): the natural language query
            num_results (Optional[int]): Number of results. Defaults to 10.
            include_domains (Optional[List(str)]): A list of top level domains to search, like ["wsj.com"]
            exclude_domains (Optional[List(str)]): Top level domains to exclude.
            start_published_date (Optional[str]): A date string like "2020-06-15".
            end_published_date (Optional[str]): End date string

        """
        response = self.client.search(
            query,
            num_results=num_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
            use_autoprompt=True,
        )
        if self._verbose:
            print(f"[Metaphor Tool] Autoprompt: {response.autoprompt_string}")
        ids = [result.id for result in response.results]
        documents = self.client.get_contents(ids)
        return [Document(text=document.extract) for document in documents.contents]

    def current_date(self):
        """
        A function to return todays date.
        Call this before any other functions that take timestamps as an argument.
        """
        return datetime.date.today()
