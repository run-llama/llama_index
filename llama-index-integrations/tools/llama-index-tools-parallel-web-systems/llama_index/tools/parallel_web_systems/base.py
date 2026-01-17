"""Parallel AI tool spec"""

from typing import Any, Dict, List, Optional, Union

import httpx

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ParallelWebSystemsToolSpec(BaseToolSpec):
    """
    Parallel AI tool spec

    This tool provides access to Parallel Web Systems Search and Extract
    APIs, enabling LLM agents to perform web research and content extraction.

    The Search API returns structured, compressed excerpts from web search
    results optimized for LLM consumption.

    The Extract API converts public URLs into clean, LLM-optimized markdown,
    including JavaScript-heavy pages and PDFs.
    """

    spec_functions = [
        "search",
        "extract",
    ]

    def __init__(self, api_key: str, base_url: Optional[str] = None) -> None:
        """
        Initialize with parameters

        Args:
            api_key: Your Parallel AI API key from https://platform.parallel.ai/
            base_url: Optional custom base URL for the API

        """
        self.api_key = api_key
        self.base_url = base_url or "https://api.parallel.ai"

    async def search(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        max_results: int = 10,
        mode: Optional[str] = None,
        excerpts: Optional[Dict[str, Any]] = None,
        source_policy: Optional[Dict[str, Any]] = None,
        fetch_policy: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search the web using Parallel Search API

        Returns structured, compressed excerpts optimized for LLM consumption.
        At least one of `objective` or `search_queries` must be provided.

        Args:
            objective: Natural-language description of what the web search is
                trying to find. This can include guidance about preferred sources
                or freshness.
            search_queries: Optional list of traditional keyword search queries
                to guide the search. May contain search operators. Max 5 queries,
                200 chars each.
            max_results: Upper bound on the number of results to return (1-40).
                The default is 10
            mode: Search mode preset. 'one-shot' returns more comprehensive results
                and longer excerpts for single response answers. 'agentic' returns
                more concise, token-efficient results for use in an agentic loop.
            excerpts: Optional settings to configure excerpt generation.
                Example: {'max_chars_per_result': 1500}
            source_policy: Optional source policy governing domain and date
                preferences in search results.
            fetch_policy: Policy for cached vs live content.
                Example: {'max_age_seconds': 86400, 'timeout_seconds': 60}

        Returns:
            A list of Document objects containing search results with excerpts
            and metadata including url, title, and publish_date.

        """
        if not objective and not search_queries:
            raise ValueError(
                "At least one of 'objective' or 'search_queries' must be provided"
            )

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "parallel-beta": "search-extract-2025-10-10",
        }

        payload: Dict[str, Any] = {
            "max_results": max_results,
        }

        if objective:
            payload["objective"] = objective
        if search_queries:
            payload["search_queries"] = search_queries
        if mode:
            payload["mode"] = mode
        if excerpts:
            payload["excerpts"] = excerpts
        if source_policy:
            payload["source_policy"] = source_policy
        if fetch_policy:
            payload["fetch_policy"] = fetch_policy

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1beta/search",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

            documents = []
            for result in data.get("results", []):
                # combine excerpts into the document text
                excerpts_list = result.get("excerpts", [])
                text = "\n\n".join(excerpts_list) if excerpts_list else ""

                doc = Document(
                    text=text,
                    metadata={
                        "url": result.get("url"),
                        "title": result.get("title"),
                        "publish_date": result.get("publish_date"),
                        "search_id": data.get("search_id"),
                    },
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error calling Parallel AI Search API: {e}")
            return []

    async def extract(
        self,
        urls: List[str],
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        excerpts: Union[bool, Dict[str, Any]] = True,
        full_content: Union[bool, Dict[str, Any]] = False,
        fetch_policy: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Extract clean, structured content from web pages using Parallel AI's Extract API.

        Converts public URLs into clean, LLM-optimized markdown including
        JavaScript-heavy pages and PDFs

        Args:
            urls: List of URLs to extract content from.
            objective: Natural language objective to focus extraction on specific
                topics. The returned excerpts will be relevant to this objective.
            search_queries: Specific keyword queries to focus extraction.
            excerpts: Include relevant excerpts. Can be True/False or a dict with
                settings like {'max_chars_per_result': 2000}. Excerpts are focused
                on objective/queries if provided.
            full_content: Include full page content. Can be True/False or a dict
                with settings like {'max_chars_per_result': 3000}.
            fetch_policy: Cache vs live content policy.
                Example: {'max_age_seconds': 86400, 'timeout_seconds': 60,
                         'disable_cache_fallback': False}

        Returns:
            A list of Document objects containing extracted content with metadata
            including url, title, publish_date, and excerpts

        """
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "parallel-beta": "search-extract-2025-10-10",
        }

        payload: Dict[str, Any] = {
            "urls": urls,
            "excerpts": excerpts,
            "full_content": full_content,
        }

        if objective:
            payload["objective"] = objective
        if search_queries:
            payload["search_queries"] = search_queries
        if fetch_policy:
            payload["fetch_policy"] = fetch_policy

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1beta/extract",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

            documents = []
            for result in data.get("results", []):
                # Use full_content if available, otherwise combine excerpts
                full_text = result.get("full_content")
                excerpts_list = result.get("excerpts", [])

                if full_text:
                    text = full_text
                elif excerpts_list:
                    text = "\n\n".join(excerpts_list)
                else:
                    text = ""

                doc = Document(
                    text=text,
                    metadata={
                        "url": result.get("url"),
                        "title": result.get("title"),
                        "publish_date": result.get("publish_date"),
                        "extract_id": data.get("extract_id"),
                        "excerpts": excerpts_list,
                    },
                )
                documents.append(doc)

            # handle any errors in response
            for error in data.get("errors", []):
                doc = Document(
                    text=f"Error extracting content: {error.get('content', 'Unknown error')}",
                    metadata={
                        "url": error.get("url"),
                        "error_type": error.get("error_type"),
                        "extract_id": data.get("extract_id"),
                    },
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error calling Parallel AI Extract API: {e}")
            return []
