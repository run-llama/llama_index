# Copyright Hewlett Packard Enterprise Development LP.

from pydi_client import DIClient
from typing import Any, Dict, Union

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode


class AlletraX10000Retriever(BaseRetriever):
    def __init__(
        self,
        uri: str,
        s3_access_key: str,
        s3_secret_key: str,
        collection_name: str,
        search_config: Union[Any, Dict[str, Any]] = None,
        top_k: int = 5,
    ):
        self.uri = uri
        self.top_k = top_k
        self.collection_name = collection_name
        self.access_key = s3_access_key
        self.secret_key = s3_secret_key
        self.search_config = search_config

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str

        client = DIClient(uri=self.uri)
        data = client.similarity_search(
            collection_name=self.collection_name,
            query=query,
            top_k=self.top_k,
            access_key=self.access_key,
            secret_key=self.secret_key,
            search_parameters=self.search_config,
        )

        nodes = []
        for item in data:
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=item["dataChunk"], metadata=item["chunkMetadata"]
                    ),
                    score=item["score"],
                )
            )
        return nodes
