"""
Vertex AI Search Retriever.

Vertex AI Search helps developers build secure, Google-quality search experiences for websites,
intranet and RAG systems for generative AI agents and apps.
Vertex AI Search is a part of Vertex AI Agent Builder.

"""

from __future__ import annotations
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InvalidArgument
from google.protobuf.json_format import MessageToDict

from llama_index.retrievers.vertexai_search._utils import get_client_info

if TYPE_CHECKING:
    from google.cloud.discoveryengine_v1beta import (
        SearchRequest,
        SearchResult,
        SearchServiceClient,
    )


class VertexAISearchRetriever(BaseRetriever):
    """
    `Vertex AI Search` retrieval.

    For a detailed explanation of the Vertex AI Search concepts
    and configuration parameters, refer to the product documentation.
    https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction

    Args:
        project_id: str
        #Google Cloud Project ID

        data_store_id: str
        #Vertex AI Search data store ID.

        location_id: str = "global"
        #Vertex AI Search data store location.

        serving_config_id: str = "default_config"
        #Vertex AI Search serving config ID

        credentials: Any = None
        The default custom credentials (google.auth.credentials.Credentials) to use
        when making API calls. If not provided, credentials will be ascertained from
        the environment

        engine_data_type: int = 0
        Defines the Vertex AI Search data type
        0 - Unstructured data
        1 - Structured data
        2 - Website data

    Example:
        retriever = VertexAISearchRetriever(
            project_id=PROJECT_ID,
            data_store_id=DATA_STORE_ID,
            location_id=LOCATION_ID,
            engine_data_type=0
        )

    """

    """
    The following parameter explanation can be found here:
    https://cloud.google.com/generative-ai-app-builder/docs/reference/rpc/google.cloud.discoveryengine.v1#contentsearchspec
    """
    filter: Optional[str] = None
    """Filter expression."""
    get_extractive_answers: bool = False
    """If True return Extractive Answers, otherwise return Extractive Segments or Snippets."""
    max_documents: int = 5
    """The maximum number of documents to return."""
    max_extractive_answer_count: int = 1
    """The maximum number of extractive answers returned in each search result.
    At most 5 answers will be returned for each SearchResult.
    """
    max_extractive_segment_count: int = 1
    """The maximum number of extractive segments returned in each search result.
    Currently one segment will be returned for each SearchResult.
    """
    query_expansion_condition: int = 1
    """Specification to determine under which conditions query expansion should occur.
    0 - Unspecified query expansion condition. In this case, server behavior defaults
        to disabled
    1 - Disabled query expansion. Only the exact search query is used, even if
        SearchResponse.total_size is zero.
    2 - Automatic query expansion built by the Search API.
    """
    spell_correction_mode: int = 1
    """Specification to determine under which conditions query expansion should occur.
    0 - Unspecified spell correction mode. In this case, server behavior defaults
        to auto.
    1 - Suggestion only. Search API will try to find a spell suggestion if there is any
        and put in the `SearchResponse.corrected_query`.
        The spell suggestion will not be used as the search query.
    2 - Automatic spell correction built by the Search API.
        Search will be based on the corrected query if found.
    """
    boost_spec: Optional[Dict[Any, Any]] = None
    """BoostSpec for boosting search results. A protobuf should be provided.
    https://cloud.google.com/generative-ai-app-builder/docs/boost-search-results
    https://cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1beta/BoostSpec
    """
    return_extractive_segment_score: bool = True
    """
    Specifies whether to return the confidence score from the extractive segments in each search result.
    This feature is available only for new or allowlisted data stores.
    """

    _client: SearchServiceClient
    _serving_config: str

    def __init__(
        self,
        project_id: str,
        data_store_id: str,
        location_id: str = "global",
        serving_config_id: str = "default_config",
        credentials: Any = None,
        engine_data_type: int = 0,
        max_documents: int = 5,
        user_agent: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes private fields."""
        self.project_id = project_id
        self.location_id = location_id
        self.data_store_id = data_store_id
        self.serving_config_id = serving_config_id
        self.engine_data_type = engine_data_type
        self.credentials = credentials
        self.max_documents = max_documents
        self._user_agent = user_agent or "llama-index/0.0.0"

        self.client_options = ClientOptions(
            api_endpoint=(
                f"{self.location_id}-discoveryengine.googleapis.com"
                if self.location_id != "global"
                else None
            )
        )

        try:
            from google.cloud.discoveryengine_v1beta import SearchServiceClient
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-discoveryengine python package. "
                "Please, install vertexaisearch dependency group: "
            ) from exc

        try:
            super().__init__(**kwargs)
        except ValueError as e:
            print(f"Error initializing GoogleVertexAISearchRetriever: {e!s}")
            raise

        #  For more information, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store

        self._client = SearchServiceClient(
            credentials=self.credentials,
            client_options=self.client_options,
            client_info=get_client_info(module="vertex-ai-search"),
        )

        self._serving_config = self._client.serving_config_path(
            project=self.project_id,
            location=self.location_id,
            data_store=self.data_store_id,
            serving_config=self.serving_config_id,
        )

    def _get_content_spec_kwargs(self) -> Optional[Dict[str, Any]]:
        """Prepares a ContentSpec object."""
        from google.cloud.discoveryengine_v1beta import SearchRequest

        if self.engine_data_type == 0:
            if self.get_extractive_answers:
                extractive_content_spec = SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_answer_count=self.max_extractive_answer_count,
                    return_extractive_segment_score=self.return_extractive_segment_score,
                )
            else:
                extractive_content_spec = SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_segment_count=self.max_extractive_segment_count,
                    return_extractive_segment_score=self.return_extractive_segment_score,
                )
            content_search_spec = {"extractive_content_spec": extractive_content_spec}
        elif self.engine_data_type == 1:
            content_search_spec = None
        elif self.engine_data_type == 2:
            content_search_spec = {
                "extractive_content_spec": SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_segment_count=self.max_extractive_segment_count,
                    max_extractive_answer_count=self.max_extractive_answer_count,
                    return_extractive_segment_score=self.return_extractive_segment_score,
                ),
                "snippet_spec": SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True
                ),
            }
        else:
            raise NotImplementedError(
                "Only data store type 0 (Unstructured), 1 (Structured),"
                "or 2 (Website) are supported currently."
                + f" Got {self.engine_data_type}"
            )
        return content_search_spec

    def _create_search_request(self, query: str) -> SearchRequest:
        """Prepares a SearchRequest object."""
        from google.cloud.discoveryengine_v1beta import SearchRequest

        query_expansion_spec = SearchRequest.QueryExpansionSpec(
            condition=self.query_expansion_condition,
        )

        spell_correction_spec = SearchRequest.SpellCorrectionSpec(
            mode=self.spell_correction_mode
        )

        content_search_spec_kwargs = self._get_content_spec_kwargs()

        if content_search_spec_kwargs is not None:
            content_search_spec = SearchRequest.ContentSearchSpec(
                **content_search_spec_kwargs
            )
        else:
            content_search_spec = None

        return SearchRequest(
            query=query,
            filter=self.filter,
            serving_config=self._serving_config,
            page_size=self.max_documents,
            content_search_spec=content_search_spec,
            query_expansion_spec=query_expansion_spec,
            spell_correction_spec=spell_correction_spec,
            boost_spec=SearchRequest.BoostSpec(**self.boost_spec)
            if self.boost_spec
            else None,
        )

    def _convert_structured_datastore_response(
        self, results: Sequence[SearchResult]
    ) -> List[NodeWithScore]:
        """Converts a sequence of search results to a list of Llamaindex note_with_score."""
        note_with_score: List[NodeWithScore] = []

        for i, result in enumerate(results):
            # Structured datastore does not have relevance score. The results are ranked
            # in order. score is calculated by below. Index 0 has the highest score
            score = (len(results) - i) / len(results)

            document_dict = MessageToDict(
                result.document._pb, preserving_proto_field_name=True
            )
            note_with_score.append(
                NodeWithScore(
                    node=TextNode(
                        text=json.dumps(document_dict.get("struct_data", {}))
                    ),
                    score=score,
                )
            )

        return note_with_score

    def _convert_unstructured_datastore_response(
        self, results: Sequence[SearchResult], chunk_type: str
    ) -> List[NodeWithScore]:
        """Converts a sequence of search results to a list of LLamaindex note_with_score."""
        note_with_score: List[NodeWithScore] = []

        for result in results:
            document_dict = MessageToDict(
                result.document._pb, preserving_proto_field_name=True
            )
            derived_struct_data = document_dict.get("derived_struct_data")
            if not derived_struct_data:
                continue

            if chunk_type not in derived_struct_data:
                continue

            for chunk in derived_struct_data[chunk_type]:
                score = chunk.get("relevanceScore", 0)
                note_with_score.append(
                    NodeWithScore(
                        node=TextNode(text=chunk.get("content", "")),
                        score=score,
                    )
                )

        return note_with_score

    def _convert_website_datastore_response(
        self, results: Sequence[SearchResult], chunk_type: str
    ) -> List[NodeWithScore]:
        """Converts a sequence of search results to a list of LLamaindex note_with_score."""
        note_with_score: List[NodeWithScore] = []

        for result in results:
            document_dict = MessageToDict(
                result.document._pb, preserving_proto_field_name=True
            )

            derived_struct_data = document_dict.get("derived_struct_data")
            if not derived_struct_data:
                continue

            if chunk_type not in derived_struct_data:
                continue

            text_field = "snippet" if chunk_type == "snippets" else "content"

            for chunk in derived_struct_data[chunk_type]:
                score = chunk.get("relevanceScore", 0)
                note_with_score.append(
                    NodeWithScore(
                        node=TextNode(text=chunk.get(text_field, "")),
                        score=score,
                    )
                )

        if not note_with_score:
            print(f"No {chunk_type} could be found.")
            if chunk_type == "extractive_answers":
                print(
                    "Make sure that your data store is using Advanced Website "
                    "Indexing.\n"
                    "https://cloud.google.com/generative-ai-app-builder/docs/about-advanced-features#advanced-website-indexing"
                )

        return note_with_score

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve from the platform."""
        """Get note_with_score relevant for a query."""

        search_request = self._create_search_request(query_bundle.query_str)

        try:
            response = self._client.search(search_request)
        except InvalidArgument as exc:
            raise type(exc)(
                exc.message
                + " This might be due to engine_data_type not set correctly."
            )

        if self.engine_data_type == 0:
            chunk_type = (
                "extractive_answers"
                if self.get_extractive_answers
                else "extractive_segments"
            )
            note_with_score = self._convert_unstructured_datastore_response(
                response.results, chunk_type
            )
        elif self.engine_data_type == 1:
            note_with_score = self._convert_structured_datastore_response(
                response.results
            )
        elif self.engine_data_type == 2:
            chunk_type = (
                "extractive_answers"
                if self.get_extractive_answers
                else "extractive_segments"
            )
            note_with_score = self._convert_website_datastore_response(
                response.results, chunk_type
            )
        else:
            raise NotImplementedError(
                "Only data store type 0 (Unstructured), 1 (Structured),"
                "or 2 (Website) are supported currently."
                + f" Got {self.engine_data_type}"
            )

        return note_with_score

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve from the platform."""
        return self._retrieve(query_bundle=query_bundle)
