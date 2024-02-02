"""Qdrant reader."""

from typing import Dict, List, Optional, cast

from llama_index.readers.base import BaseReader
from llama_index.schema import Document


class QdrantReader(BaseReader):
    """Qdrant reader.

    Retrieve documents from existing Qdrant collections.

    Args:
        location:
            If `:memory:` - use in-memory Qdrant instance.
            If `str` - use it as a `url` parameter.
            If `None` - use default values for `host` and `port`.
        url:
            either host or str of
            "Optional[scheme], host, Optional[port], Optional[prefix]".
            Default: `None`
        port: Port of the REST API interface. Default: 6333
        grpc_port: Port of the gRPC interface. Default: 6334
        prefer_grpc: If `true` - use gPRC interface whenever possible in custom methods.
        https: If `true` - use HTTPS(SSL) protocol. Default: `false`
        api_key: API key for authentication in Qdrant Cloud. Default: `None`
        prefix:
            If not `None` - add `prefix` to the REST URL path.
            Example: `service/v1` will result in
            `http://localhost:6333/service/v1/{qdrant-endpoint}` for REST API.
            Default: `None`
        timeout:
            Timeout for REST and gRPC API requests.
            Default: 5.0 seconds for REST and unlimited for gRPC
        host: Host name of Qdrant service. If url and host are None, set to 'localhost'.
            Default: `None`
    """

    def __init__(
        self,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
    ):
        """Initialize with parameters."""
        import_err_msg = (
            "`qdrant-client` package not found, please run `pip install qdrant-client`"
        )
        try:
            import qdrant_client
        except ImportError:
            raise ImportError(import_err_msg)

        self._client = qdrant_client.QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
        )

    def load_data(
        self,
        collection_name: str,
        query_vector: List[float],
        should_search_mapping: Optional[Dict[str, str]] = None,
        must_search_mapping: Optional[Dict[str, str]] = None,
        must_not_search_mapping: Optional[Dict[str, str]] = None,
        rang_search_mapping: Optional[Dict[str, Dict[str, float]]] = None,
        limit: int = 10,
    ) -> List[Document]:
        """Load data from Qdrant.

        Args:
            collection_name (str): Name of the Qdrant collection.
            query_vector (List[float]): Query vector.
            should_search_mapping (Optional[Dict[str, str]]): Mapping from field name
                to query string.
            must_search_mapping (Optional[Dict[str, str]]): Mapping from field name
                to query string.
            must_not_search_mapping (Optional[Dict[str, str]]): Mapping from field
                name to query string.
            rang_search_mapping (Optional[Dict[str, Dict[str, float]]]): Mapping from
                field name to range query.
            limit (int): Number of results to return.

        Example:
            reader = QdrantReader()
            reader.load_data(
                 collection_name="test_collection",
                 query_vector=[0.1, 0.2, 0.3],
                 should_search_mapping={"text_field": "text"},
                 must_search_mapping={"text_field": "text"},
                 must_not_search_mapping={"text_field": "text"},
                 # gte, lte, gt, lt supported
                 rang_search_mapping={"text_field": {"gte": 0.1, "lte": 0.2}},
                 limit=10
            )

        Returns:
            List[Document]: A list of documents.
        """
        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchText,
            MatchValue,
            Range,
        )
        from qdrant_client.http.models.models import Payload

        should_search_mapping = should_search_mapping or {}
        must_search_mapping = must_search_mapping or {}
        must_not_search_mapping = must_not_search_mapping or {}
        rang_search_mapping = rang_search_mapping or {}

        should_search_conditions = [
            FieldCondition(key=key, match=MatchText(text=value))
            for key, value in should_search_mapping.items()
            if should_search_mapping
        ]
        must_search_conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in must_search_mapping.items()
            if must_search_mapping
        ]
        must_not_search_conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in must_not_search_mapping.items()
            if must_not_search_mapping
        ]
        rang_search_conditions = [
            FieldCondition(
                key=key,
                range=Range(
                    gte=value.get("gte"),
                    lte=value.get("lte"),
                    gt=value.get("gt"),
                    lt=value.get("lt"),
                ),
            )
            for key, value in rang_search_mapping.items()
            if rang_search_mapping
        ]
        should_search_conditions.extend(rang_search_conditions)
        response = self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=must_search_conditions,
                must_not=must_not_search_conditions,
                should=should_search_conditions,
            ),
            with_vectors=True,
            with_payload=True,
            limit=limit,
        )

        documents = []
        for point in response:
            payload = cast(Payload, point.payload)
            try:
                vector = cast(List[float], point.vector)
            except ValueError as e:
                raise ValueError("Could not cast vector to List[float].") from e
            document = Document(
                id_=payload.get("doc_id"),
                text=payload.get("text"),
                metadata=payload.get("metadata"),
                embedding=vector,
            )
            documents.append(document)

        return documents
