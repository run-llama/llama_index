from typing import List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.core import BaseRetriever
from llama_index.indices.query.query_transform.base import BaseQueryTransform
from llama_index.prompts.mixin import PromptMixinType
from llama_index.schema import NodeWithScore, QueryBundle


class TransformRetriever(BaseRetriever):
    """Transform Retriever.

    Takes in an existing retriever and a query transform and runs the query transform
    before running the retriever.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        query_transform: BaseQueryTransform,
        transform_metadata: Optional[dict] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._query_transform = query_transform
        self._transform_metadata = transform_metadata
        super().__init__(callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        # NOTE: don't include tools for now
        return {"query_transform": self._query_transform}

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_bundle = self._query_transform.run(
            query_bundle, metadata=self._transform_metadata
        )
        return self._retriever.retrieve(query_bundle)
