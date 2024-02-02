import logging
from typing import List, Optional

import requests

from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.indices.managed.zilliz.base import ZillizCloudPipelineIndex
from llama_index.legacy.indices.query.schema import QueryBundle
from llama_index.legacy.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.legacy.vector_stores.types import MetadataFilters

logger = logging.getLogger(__name__)


class ZillizCloudPipelineRetriever(BaseRetriever):
    """A retriever built on top of Zilliz Cloud Pipeline's index."""

    def __init__(
        self,
        index: ZillizCloudPipelineIndex,
        search_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        filters: Optional[MetadataFilters] = None,
        offset: int = 0,
        output_metadata: list = [],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self.search_top_k = search_top_k
        if filters:
            exprs = []
            for fil in filters.filters:
                expr = f"{fil.key} == '{fil.value}'"
                exprs.append(expr)
            self.filter = " && ".join(exprs)
        else:
            self.filter = ""
        self.offset = offset

        search_pipe_id = index.pipeline_ids.get("SEARCH")
        self.search_pipeline_url = f"{index.domain}/{search_pipe_id}/run"
        self.headers = index.headers
        self.output_fields = output_metadata
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        params = {
            "data": {"query_text": query_bundle.query_str},
            "params": {
                "limit": self.search_top_k,
                "offset": self.offset,
                "outputFields": ["chunk_text", *self.output_fields],
                "filter": self.filter,
            },
        }

        response = requests.post(
            self.search_pipeline_url, headers=self.headers, json=params
        )
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        response_data = response_dict["data"]

        top_nodes = []
        for search_res in response_data["result"]:
            text = search_res.pop("chunk_text")
            entity_id = search_res.pop("id")
            distance = search_res.pop("distance")
            node = NodeWithScore(
                node=TextNode(text=text, id_=entity_id, metadata=search_res),
                score=distance,
            )
            top_nodes.append(node)
        return top_nodes
