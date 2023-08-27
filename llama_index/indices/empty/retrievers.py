"""Default query for EmptyIndex."""
from typing import Any, List, Optional

from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.empty.base import EmptyIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.schema import NodeWithScore


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
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._input_prompt = input_prompt or DEFAULT_SIMPLE_INPUT_PROMPT

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve relevant nodes."""
        del query_bundle  # Unused
        return []
