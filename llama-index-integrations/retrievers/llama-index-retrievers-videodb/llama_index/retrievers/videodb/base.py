from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

import logging
import os

from typing import List, Optional
from videodb import connect
from videodb import SearchType, IndexType


logger = logging.getLogger(__name__)


class VideoDBRetriever(BaseRetriever):
    def __init__(
        self,
        api_key: Optional[str] = None,
        collection: Optional[str] = "default",
        video: Optional[str] = None,
        score_threshold: Optional[float] = 0.2,
        result_threshold: Optional[int] = 5,
        search_type: Optional[str] = SearchType.semantic,
        index_type: Optional[str] = IndexType.spoken_word,
        scene_index_id: Optional[str] = None,
        base_url: Optional[str] = None,
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
        self.scene_index_id = scene_index_id
        self.index_type = index_type
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        kwargs = {"api_key": self._api_key}
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        conn = connect(**kwargs)
        if self.video:
            search_args = {
                "query": query_bundle.query_str,
                "search_type": self.search_type,
                "index_type": self.index_type,
                "score_threshold": self.score_threshold,
                "result_threshold": self.result_threshold,
            }
            if self.index_type == IndexType.scene and self.scene_index_id:
                search_args["index_id"] = self.scene_index_id
            coll = conn.get_collection(self.collection)
            video = coll.get_video(self.video)
            search_res = video.search(**search_args)
        else:
            coll = conn.get_collection(self.collection)
            search_res = coll.search(
                query_bundle.query_str,
                search_type=self.search_type,
                index_type=self.index_type,
                score_threshold=self.score_threshold,
                result_threshold=self.result_threshold,
            )

        nodes = []
        collection_id = search_res.collection_id
        for shot in search_res.get_shots():
            score = shot.search_score
            textnode = TextNode(
                text=shot.text,
                metadata={
                    "collection_id": collection_id,
                    "video_id": shot.video_id,
                    "length": shot.video_length,
                    "title": shot.video_title,
                    "start": shot.start,
                    "end": shot.end,
                    "type": self.index_type,
                },
            )
            nodes.append(NodeWithScore(node=textnode, score=score))
        return nodes
