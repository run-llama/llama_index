import asyncio
import logging
from typing import Any, Dict, List, Optional

import requests
from llama_index.core import get_response_synthesizer
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import (
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

API_ENDPOINT = "https://api.trytldw.ai/v1"


class Fragment(BaseModel):
    """Represents a fragment of a video scene with metadata."""

    uuid: str
    start_ms: float
    end_ms: float
    similarity: float
    description: str


class Scene(BaseModel):
    """Represents a video scene containing multiple fragments."""

    media_id: str
    external_id: str
    start_ms: float
    end_ms: float
    max_similarity: float
    fragments: List[Fragment]


class SearchResult(BaseModel):
    """Encapsulates the search results from the TL;DW API."""

    scenes: List[Scene]
    metadata: Dict[str, Any]


class TldwRetriever(BaseRetriever):
    r"""
    A retriever that searches for relevant video moments from the TL;DW collection.

    Args:
        api_key (str): The API key for authentication.
        collection_id (str): The ID of the video collection to search within.
        return_fragments (bool): Whether to return individual fragments or summarized scenes.
        scene_summerizer (BaseSynthesizer): Synthesizer to summarize scenes if return_fragments is False.
        callback_manager (Optional[CallbackManager]): Optional callback manager for logging and event handling.
    """

    def __init__(
        self,
        api_key: str,
        collection_id: str,
        return_fragments: bool = True,
        scene_summerizer: BaseSynthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT
        ),
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._api_key = api_key
        self._collection_id = collection_id
        self._return_fragments = return_fragments
        self._scene_summerizer = scene_summerizer
        super().__init__(
            callback_manager=callback_manager,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        res = requests.post(
            f"{API_ENDPOINT}/search",
            headers=headers,
            json={
                "collection_id": self._collection_id,
                "search_term": query_bundle.query_str,
            },
        )
        search_results = SearchResult.model_validate(res.json())

        if self._return_fragments:
            # Return individual fragments as nodes
            return [
                NodeWithScore(
                    node=TextNode(
                        text=fragment.description,
                        metadata={
                            "collection_id": self._collection_id,
                            "media_id": scene.media_id,
                            "start_ms": fragment.start_ms,
                            "end_ms": fragment.end_ms,
                            "scene_start_ms": scene.start_ms,
                            "scene_end_ms": scene.end_ms,
                        },
                    ),
                    score=fragment.similarity,
                )
                for scene in search_results.scenes
                for fragment in scene.fragments
            ]
        else:
            # Summarize scenes and return them as nodes asynchronously
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                asyncio.gather(
                    *(
                        self._asummarize_scene(scene, query_bundle.query_str)
                        for scene in search_results.scenes
                    )
                )
            )

    async def _asummarize_scene(self, scene: Scene, query_str: str) -> NodeWithScore:
        scene_summary = await self._scene_summerizer.aget_response(
            query_str,
            [fragment.description for fragment in scene.fragments],
        )
        return NodeWithScore(
            node=TextNode(
                text=scene_summary,
                metadata={
                    "collection_id": self._collection_id,
                    "media_id": scene.media_id,
                    "start_ms": scene.start_ms,
                    "end_ms": scene.end_ms,
                },
            ),
            score=scene.max_similarity,
        )
