from __future__ import annotations

from typing import Any, List, Optional

try:
    # LlamaIndex v0.11+ layout
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
except Exception:  # pragma: no cover - test import fallback
    # Fallback names to allow tests to import without the monorepo
    class BaseRetriever:  # type: ignore
        def __init__(self, **_: Any) -> None:
            pass

    class QueryBundle:  # type: ignore
        def __init__(self, query_str: str) -> None:
            self.query_str = query_str

    class TextNode:  # type: ignore
        def __init__(self, text: str, metadata: Optional[dict] = None) -> None:
            self.text = text
            self.metadata = metadata or {}

    class NodeWithScore:  # type: ignore
        def __init__(self, node: TextNode, score: float = 1.0) -> None:
            self.node = node
            self.score = score


class SuperlinkedRetriever(BaseRetriever):
    """
    LlamaIndex retriever for Superlinked.

    Parameters
    ----------
    sl_client : Any
        An instance of a Superlinked App.
    sl_query : Any
        A Superlinked QueryDescriptor object.
    page_content_field : str
        Name of the field in Superlinked result to expose as node text.
    query_text_param : str, default "query_text"
        Parameter name in the Superlinked query for user text.
    metadata_fields : Optional[List[str]]
        If None, include all fields except `page_content_field`.
        Otherwise include only the specified fields.
    k : int, default 4
        Max number of nodes returned (final cap applied client-side).

    """

    def __init__(
        self,
        *,
        sl_client: Any,
        sl_query: Any,
        page_content_field: str,
        query_text_param: str = "query_text",
        metadata_fields: Optional[List[str]] = None,
        k: int = 4,
        callback_manager: Optional[Any] = None,
    ) -> None:
        # Import and validate types lazily to avoid hard dependency at import time
        try:
            from superlinked.framework.dsl.app.app import App  # type: ignore
            from superlinked.framework.dsl.query.query_descriptor import (  # type: ignore
                QueryDescriptor,
            )
        except Exception as exc:  # pragma: no cover - exercised in unit tests via mocks
            raise ImportError(
                "The 'superlinked' package is required. Install with 'pip install superlinked'"
            ) from exc

        if not isinstance(sl_client, App):
            raise TypeError("sl_client must be a Superlinked App instance")
        if not isinstance(sl_query, QueryDescriptor):
            raise TypeError("sl_query must be a Superlinked QueryDescriptor instance")

        self.sl_client = sl_client
        self.sl_query = sl_query
        self.page_content_field = page_content_field
        self.query_text_param = query_text_param
        self.metadata_fields = metadata_fields
        self.k = k

        # Initialize BaseRetriever
        try:
            super().__init__(callback_manager)
        except TypeError:
            # Fallback BaseRetriever signature in tests accepts **kwargs
            super().__init__()

    # LlamaIndex retrievers implement _retrieve(QueryBundle)
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:  # type: ignore[override]
        user_query = getattr(query_bundle, "query_str", str(query_bundle))

        # Build query params, allowing overrides via retriever metadata if needed later
        query_params: dict[str, Any] = {self.query_text_param: user_query}

        try:
            result = self.sl_client.query(
                query_descriptor=self.sl_query, **query_params
            )
        except Exception:
            return []

        nodes: List[NodeWithScore] = []
        for entry in getattr(result, "entries", []) or []:
            fields = getattr(entry, "fields", None) or {}
            if self.page_content_field not in fields:
                continue

            text = fields[self.page_content_field]
            metadata: dict[str, Any] = {"id": getattr(entry, "id", None)}

            if self.metadata_fields is None:
                for key, val in fields.items():
                    if key != self.page_content_field:
                        metadata[key] = val
            else:
                for key in self.metadata_fields:
                    if key in fields:
                        metadata[key] = fields[key]

            # Determine score from Superlinked metadata if available
            score_value: float = 1.0
            entry_metadata = getattr(entry, "metadata", None)
            if entry_metadata is not None and hasattr(entry_metadata, "score"):
                try:
                    score_value = float(entry_metadata.score)
                except Exception:
                    score_value = 1.0

            node = TextNode(text=text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=score_value))

        return nodes[: self.k]
