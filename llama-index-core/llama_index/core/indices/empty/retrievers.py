"""Default query for EmptyIndex."""
from typing import Any, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.empty.base import EmptyIndex
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.core.schema import NodeWithScore, QueryBundle


class EmptyIndexRetriever(BaseRetriever):
    """EmptyIndex query.

    Passes the raw LLM call to the underlying LLM model.

    Args:
        input_prompt (Optional[BasePromptTemplate]): A Simple Input Prompt
            (see :ref:`Prompt-Templates`).

    """

    def __init__(
        self,
        index: EmptyIndex,
        input_prompt: Optional[BasePromptTemplate] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._input_prompt = input_prompt or DEFAULT_SIMPLE_INPUT_PROMPT
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve relevant nodes."""
        del query_bundle  # Unused
        return []
