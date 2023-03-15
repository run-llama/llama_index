"""Node postprocessor."""

import re
from abc import abstractmethod
from typing import Dict, List, Optional, cast

from pydantic import BaseModel, Field

from gpt_index.data_structs.data_structs import Node
from gpt_index.indices.postprocessor import BasePostprocessor
from gpt_index.indices.query.embedding_utils import SimilarityTracker


class BaseNodePostprocessor(BasePostprocessor, BaseModel):
    """Node postprocessor."""

    @abstractmethod
    def postprocess_nodes(
        self, nodes: List[Node], extra_info: Optional[Dict] = None
    ) -> List[Node]:
        """Postprocess nodes."""


class KeywordNodePostprocessor(BaseNodePostprocessor):
    """Keyword-based Node processor."""

    required_keywords: List[str] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)

    def postprocess_nodes(
        self, nodes: List[Node], extra_info: Optional[Dict] = None
    ) -> List[Node]:
        """Postprocess nodes."""
        new_nodes = []
        for node in nodes:
            words = re.findall(r"\w+", node.get_text())
            should_use_node = True
            if self.required_keywords is not None:
                for w in self.required_keywords:
                    if w not in words:
                        should_use_node = False

            if self.exclude_keywords is not None:
                for w in self.exclude_keywords:
                    if w in words:
                        should_use_node = False

            if should_use_node:
                new_nodes.append(node)

        return new_nodes


class SimilarityPostprocessor(BaseNodePostprocessor):
    """Similarity-based Node processor."""

    similarity_cutoff: float = Field(default=None)

    def postprocess_nodes(
        self, nodes: List[Node], extra_info: Optional[Dict] = None
    ) -> List[Node]:
        """Postprocess nodes."""
        extra_info = extra_info or {}
        similarity_tracker = extra_info.get("similarity_tracker", None)
        if similarity_tracker is None:
            return nodes
        sim_cutoff_exists = (
            similarity_tracker is not None and self.similarity_cutoff is not None
        )

        new_nodes = []
        for node in nodes:
            should_use_node = True
            if sim_cutoff_exists:
                similarity = cast(SimilarityTracker, similarity_tracker).find(node)
                if similarity is None:
                    should_use_node = False
                if cast(float, similarity) < cast(float, self.similarity_cutoff):
                    should_use_node = False

            if should_use_node:
                new_nodes.append(node)

        return new_nodes
