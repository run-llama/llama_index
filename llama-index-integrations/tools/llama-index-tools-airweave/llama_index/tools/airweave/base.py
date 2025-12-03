"""Airweave tool spec."""

import warnings
from typing import Any, Dict, List, Optional

from airweave import AirweaveSDK, SearchRequest
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class AirweaveToolSpec(BaseToolSpec):
    """
    Airweave tool spec for searching collections.

    Airweave is an open-source platform that makes any app searchable
    for your agent by syncing data from various sources.

    To use this tool, you need:
    1. An Airweave account and API key
    2. At least one collection set up with data

    Get started at https://airweave.ai/
    """

    spec_functions = [
        "search_collection",
        "advanced_search_collection",
        "search_and_generate_answer",
        "list_collections",
        "get_collection_info",
    ]

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        framework_name: str = "llamaindex",
        framework_version: str = "0.1.0",
    ) -> None:
        """
        Initialize with Airweave API credentials.

        Args:
            api_key: Your Airweave API key from the dashboard
            base_url: Optional custom base URL for self-hosted instances
            framework_name: Framework name for analytics (default: "llamaindex")
            framework_version: Framework version for analytics

        """
        init_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "framework_name": framework_name,
            "framework_version": framework_version,
        }

        if base_url:
            init_kwargs["base_url"] = base_url

        self.client = AirweaveSDK(**init_kwargs)

    def search_collection(
        self,
        collection_id: str,
        query: str,
        limit: Optional[int] = 10,
        offset: Optional[int] = 0,
    ) -> List[Document]:
        """
        Search a specific Airweave collection with a natural language query.

        This is a simple search function for common use cases. For advanced
        options like reranking or answer generation, use advanced_search_collection.

        Args:
            collection_id: The readable ID of the collection to search
                          (e.g., 'finance-data-ab123')
            query: The search query in natural language
            limit: Maximum number of results to return (default: 10)
            offset: Number of results to skip for pagination (default: 0)

        Returns:
            List of Document objects containing search results with metadata

        """
        response = self.client.collections.search(
            readable_id=collection_id,
            request=SearchRequest(query=query, limit=limit, offset=offset),
        )

        return self._parse_search_response(response, collection_id)

    def advanced_search_collection(
        self,
        collection_id: str,
        query: str,
        limit: Optional[int] = 10,
        offset: Optional[int] = 0,
        retrieval_strategy: Optional[str] = None,
        temporal_relevance: Optional[float] = None,
        expand_query: Optional[bool] = None,
        interpret_filters: Optional[bool] = None,
        rerank: Optional[bool] = None,
        generate_answer: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Advanced search with full control over retrieval parameters.

        Args:
            collection_id: The readable ID of the collection
            query: The search query in natural language
            limit: Maximum number of results to return (default: 10)
            offset: Number of results to skip for pagination (default: 0)
            retrieval_strategy: Search strategy - "hybrid", "neural", or "keyword"
                              - hybrid: combines semantic and keyword search (default)
                              - neural: pure semantic/embedding search
                              - keyword: traditional BM25 keyword search
            temporal_relevance: Weight recent content higher (0.0-1.0)
                              0.0 = no recency bias, 1.0 = only recent matters
            expand_query: Generate query variations for better recall
            interpret_filters: Extract structured filters from natural language
            rerank: Use LLM-based reranking for improved relevance
            generate_answer: Generate a natural language answer from results

        Returns:
            Dictionary with 'documents' list and optional 'answer' field
            Example: {"documents": [...], "answer": "Generated answer text"}

        """
        search_params: Dict[str, Any] = {
            "query": query,
            "limit": limit,
            "offset": offset,
        }

        # Add optional parameters
        if retrieval_strategy:
            search_params["retrieval_strategy"] = retrieval_strategy
        if temporal_relevance is not None:
            search_params["temporal_relevance"] = temporal_relevance
        if expand_query is not None:
            search_params["expand_query"] = expand_query
        if interpret_filters is not None:
            search_params["interpret_filters"] = interpret_filters
        if rerank is not None:
            search_params["rerank"] = rerank
        if generate_answer is not None:
            search_params["generate_answer"] = generate_answer

        response = self.client.collections.search(
            readable_id=collection_id,
            request=SearchRequest(**search_params),
        )

        result: Dict[str, Any] = {
            "documents": self._parse_search_response(response, collection_id),
        }

        # Add generated answer if available
        if hasattr(response, "completion") and response.completion:
            result["answer"] = response.completion

        return result

    def search_and_generate_answer(
        self,
        collection_id: str,
        query: str,
        limit: Optional[int] = 10,
        use_reranking: bool = True,
    ) -> Optional[str]:
        """
        Search collection and generate a natural language answer (RAG-style).

        This is a convenience method that combines search with answer generation,
        perfect for agents that need direct answers rather than raw documents.

        Args:
            collection_id: The readable ID of the collection
            query: The search query / question in natural language
            limit: Maximum number of results to consider (default: 10)
            use_reranking: Whether to use LLM reranking (default: True)

        Returns:
            Natural language answer generated from the search results,
            or None if no answer could be generated (with a warning)

        """
        response = self.client.collections.search(
            readable_id=collection_id,
            request=SearchRequest(
                query=query,
                limit=limit,
                generate_answer=True,
                rerank=use_reranking,
            ),
        )

        if hasattr(response, "completion") and response.completion:
            return response.completion
        else:
            # Fallback if no answer generated
            warnings.warn(
                "No answer could be generated from the search results", UserWarning
            )
            return None

    def _parse_search_response(
        self, response: Any, collection_id: str
    ) -> List[Document]:
        """Parse Airweave search response into LlamaIndex Documents."""
        documents = []

        if hasattr(response, "results") and response.results:
            for result in response.results:
                # Extract text content
                text_content = ""
                if isinstance(result, dict):
                    text_content = (
                        result.get("content") or result.get("text") or str(result)
                    )
                elif hasattr(result, "content"):
                    text_content = result.content
                elif hasattr(result, "text"):
                    text_content = result.text
                else:
                    text_content = str(result)

                # Build metadata
                metadata: Dict[str, Any] = {"collection_id": collection_id}

                if isinstance(result, dict):
                    if "metadata" in result:
                        metadata.update(result["metadata"])
                    if "score" in result:
                        metadata["score"] = result["score"]
                    if "source" in result:
                        metadata["source"] = result["source"]
                    if "id" in result:
                        metadata["result_id"] = result["id"]
                else:
                    if hasattr(result, "metadata") and result.metadata:
                        metadata.update(result.metadata)
                    if hasattr(result, "score"):
                        metadata["score"] = result.score
                    if hasattr(result, "source"):
                        metadata["source"] = result.source
                    if hasattr(result, "id"):
                        metadata["result_id"] = result.id

                documents.append(Document(text=text_content, metadata=metadata))

        return documents

    def list_collections(
        self,
        skip: Optional[int] = 0,
        limit: Optional[int] = 100,
    ) -> List[Dict[str, Any]]:
        """
        List all collections available in your Airweave organization.

        Useful for discovering what collections are available to search.

        Args:
            skip: Number of collections to skip for pagination (default: 0)
            limit: Maximum number of collections to return, 1-1000 (default: 100)

        Returns:
            List of dictionaries with collection information

        """
        collections = self.client.collections.list(skip=skip, limit=limit)

        return [
            {
                "id": (
                    coll.readable_id if hasattr(coll, "readable_id") else str(coll.id)
                ),
                "name": coll.name,
                "created_at": (
                    str(coll.created_at) if hasattr(coll, "created_at") else None
                ),
            }
            for coll in collections
        ]

    def get_collection_info(self, collection_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific collection.

        Args:
            collection_id: The readable ID of the collection

        Returns:
            Dictionary with detailed collection information

        """
        collection = self.client.collections.get(readable_id=collection_id)

        return {
            "id": (
                collection.readable_id
                if hasattr(collection, "readable_id")
                else str(collection.id)
            ),
            "name": collection.name,
            "created_at": (
                str(collection.created_at)
                if hasattr(collection, "created_at")
                else None
            ),
            "description": getattr(collection, "description", None),
        }
