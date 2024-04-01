from typing import Any, Dict, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.labelled_property_graph.base import (
    LabelledPropertyGraphIndex,
)
from llama_index.core.indices.labelled_property_graph.sub_retrievers.base import (
    BaseLPGRetriever,
)
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle


class LPGVectorRetriever(BaseLPGRetriever):
    def __init__(
        self,
        index: LabelledPropertyGraphIndex,
        include_text: bool = True,
        embed_model: Optional[BaseEmbedding] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        retriever_kwargs = retriever_kwargs or {}
        self._retriever = VectorIndexRetriever(
            index=index, embed_model=embed_model, **retriever_kwargs
        )

        super().__init__(index=index, include_text=include_text, **kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)

        return self._parse_results(nodes)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)

        return self._parse_results(nodes)
