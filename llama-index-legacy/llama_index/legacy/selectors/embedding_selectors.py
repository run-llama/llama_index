from typing import Any, Dict, Optional, Sequence

from llama_index.legacy.core.base_selector import (
    BaseSelector,
    SelectorResult,
    SingleSelection,
)
from llama_index.legacy.embeddings.base import BaseEmbedding
from llama_index.legacy.embeddings.utils import resolve_embed_model
from llama_index.legacy.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.legacy.prompts.mixin import PromptDictType
from llama_index.legacy.schema import QueryBundle
from llama_index.legacy.tools.types import ToolMetadata


class EmbeddingSingleSelector(BaseSelector):
    """Embedding selector.

    Embedding selector that chooses one out of many options.

    Args:
        embed_model (BaseEmbedding): An embedding model.
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
    ) -> None:
        self._embed_model = embed_model

    @classmethod
    def from_defaults(
        cls,
        embed_model: Optional[BaseEmbedding] = None,
    ) -> "EmbeddingSingleSelector":
        # optionally initialize defaults
        embed_model = embed_model or resolve_embed_model("default")

        # construct prompt
        return cls(embed_model)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        query_embedding = self._embed_model.get_query_embedding(query.query_str)
        text_embeddings = [
            self._embed_model.get_text_embedding(choice.description)
            for choice in choices
        ]

        top_similarities, top_ids = get_top_k_embeddings(
            query_embedding,
            text_embeddings,
            similarity_top_k=1,
            embedding_ids=list(range(len(choices))),
        )
        # get top choice
        top_selection_reason = f"Top similarity match: {top_similarities[0]:.2f}, {choices[top_ids[0]].name}"
        top_selection = SingleSelection(index=top_ids[0], reason=top_selection_reason)

        # parse output
        return SelectorResult(selections=[top_selection])

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        query_embedding = await self._embed_model.aget_query_embedding(query.query_str)
        text_embeddings = [
            await self._embed_model.aget_text_embedding(choice.description)
            for choice in choices
        ]

        top_similarities, top_ids = get_top_k_embeddings(
            query_embedding,
            text_embeddings,
            similarity_top_k=1,
            embedding_ids=list(range(len(choices))),
        )
        # get top choice
        top_selection_reason = f"Top similarity match: {top_similarities[0]:.2f}, {choices[top_ids[0]].name}"
        top_selection = SingleSelection(index=top_ids[0], reason=top_selection_reason)

        # parse output
        return SelectorResult(selections=[top_selection])
