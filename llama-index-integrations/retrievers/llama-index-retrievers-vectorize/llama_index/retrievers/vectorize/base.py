"""Vectorize retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from vectorize_client import (
    ApiClient,
    Configuration,
    Document,
    PipelinesApi,
    RetrieveDocumentsRequest,
)

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)

if TYPE_CHECKING:
    from llama_index.core.callbacks.base import CallbackManager


class VectorizeRetriever(BaseRetriever):
    """Vectorize retriever.

    Setup:
        Install package ``llama-index-vectorize``

        .. code-block:: bash

            pip install -U llama-index-retrievers-vectorize

    Instantiate:
        .. code-block:: python

            from llama_index.retrievers.vectorize import VectorizeRetriever

            retriever = VectorizeRetriever(
                api_token="xxxxx", "organization"="1234", "pipeline_id"="5678"
            )

    Usage:
        .. code-block:: python

            query = "what year was breath of the wild released?"
            retriever.retrieve(query)

    Args:
        api_token: The Vectorize API token.
        environment: The Vectorize API environment (prod, dev, local, staging).
            Defaults to prod.
        organization: The Vectorize organization.
        pipeline_id: The Vectorize pipeline ID.
        num_results: The number of documents to return.
        rerank: Whether to rerank the retrieved documents.
        metadata_filters: The metadata filters to apply when retrieving documents.
        callback_manager: The callback manager to use for callbacks.
        verbose: Whether to enable verbose logging.
    """

    def __init__(  # noqa: D107
        self,
        api_token: str,
        *,
        environment: Literal["prod", "dev", "local", "staging"] = "prod",
        organization: str | None = None,
        pipeline_id: str | None = None,
        num_results: int = 5,
        rerank: bool = False,
        metadata_filters: list[dict[str, Any]] | None = None,
        callback_manager: CallbackManager | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(callback_manager=callback_manager, verbose=verbose)
        self.organization = organization
        self.pipeline_id = pipeline_id
        self.num_results = num_results
        self.rerank = rerank
        self.metadata_filters = metadata_filters
        header_name = None
        header_value = None
        if environment == "prod":
            host = "https://api.vectorize.io/v1"
        elif environment == "dev":
            host = "https://api-dev.vectorize.io/v1"
        elif environment == "local":
            host = "http://localhost:3000/api"
            header_name = "x-lambda-api-key"
            header_value = api_token
        else:
            host = "https://api-staging.vectorize.io/v1"
        api = ApiClient(
            Configuration(host=host, access_token=api_token, debug=True),
            header_name,
            header_value,
        )
        self._pipelines = PipelinesApi(api)

    @staticmethod
    def _convert_document(document: Document) -> NodeWithScore:
        doc = TextNode(
            id_=document.id,
            text=document.text,
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=document.unique_source)
            },
        )
        return NodeWithScore(node=doc, score=document.similarity)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str

        request = RetrieveDocumentsRequest(
            question=query,
            num_results=self.num_results,
            rerank=self.rerank,
            metadata_filters=self.metadata_filters,
        )
        response = self._pipelines.retrieve_documents(
            self.organization, self.pipeline_id, request
        )

        return [self._convert_document(doc) for doc in response.documents]
