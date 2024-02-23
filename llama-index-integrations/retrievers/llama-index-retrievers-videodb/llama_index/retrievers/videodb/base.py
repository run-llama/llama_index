from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

import requests
import logging
import os

from typing import List, Optional


logger = logging.getLogger(__name__)


class SearchType:
    semantic = "semantic"


VIDEODB_BASE_URL = "https://api.videodb.io"


class VideoDBRetriever(BaseRetriever):
    def __init__(
        self,
        api_key: Optional[str] = None,
        collection: Optional[str] = "default",
        video: Optional[str] = None,
        score_threshold: Optional[float] = 0.2,
        result_threshold: Optional[int] = 5,
        search_type: Optional[str] = SearchType.semantic,
        base_url: Optional[str] = VIDEODB_BASE_URL,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Creates a new VideoDB Retriever."""
        if api_key is None:
            api_key = os.environ.get("VIDEO_DB_API_KEY")
        if api_key is None:
            raise Exception(
                "No API key provided. Set an API key either as an environment variable (VIDEO_DB_API_KEY) or pass it as an argument."
            )
        self._api_key = api_key
        self._base_url = base_url
        self.video = video
        self.collection = collection
        self.score_threshold = score_threshold
        self.result_threshold = result_threshold
        self.search_type = search_type
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        if self.video:
            path = f"video/{self.video}/search"
        else:
            path = f"collection/{self.collection}/search"
        url = f"{self._base_url}/{path}"

        headers = {"x-access-token": self._api_key, "Content-Type": "application/json"}

        payload = {
            "type": self.search_type,
            "query": query_bundle.query_str,
            "score_threshold": self.score_threshold,
            "result_threshold": self.result_threshold,
        }

        res = requests.post(url, headers=headers, json=payload).json()
        if res.get("success"):
            search_res = res.get("data")
        else:
            logger.error(f"Error in VideoDB Retrieval: {res.get('mesage')}")
            raise Exception(f"Error in VideoDB Retrieval: {res.get('message')}")

        nodes = []
        results = search_res.get("results", [])
        for result in results:
            collection_id = result.get("collection_id")
            video_id = result.get("video_id")
            length = result.get("length")
            title = result.get("title")
            docs = result.get("docs", [])
            for doc in docs:
                textnode = TextNode(
                    text=doc.get("text"),
                    metadata={
                        "collection_id": collection_id,
                        "video_id": video_id,
                        "length": length,
                        "title": title,
                        "start": doc.get("start"),
                        "end": doc.get("end"),
                    },
                )
                score = doc.get("score", 0)
                nodes.append(NodeWithScore(node=textnode, score=score))
        return nodes
