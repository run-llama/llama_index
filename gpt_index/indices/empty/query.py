"""Default query for GPTEmptyIndex."""
from typing import Any, List, Optional

from gpt_index.data_structs.data_structs_v2 import EmptyIndex
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.data_structs.node_v2 import NodeWithScore
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.type import ResponseMode
from gpt_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from gpt_index.prompts.prompts import SimpleInputPrompt


class GPTEmptyIndexQuery(BaseGPTIndexQuery[EmptyIndex]):
    """GPTEmptyIndex query.

    Passes the raw LLM call to the underlying LLM model.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    Args:
        input_prompt (Optional[SimpleInputPrompt]): A Simple Input Prompt
            (see :ref:`Prompt-Templates`).

    """

    def __init__(
        self,
        input_prompt: Optional[SimpleInputPrompt] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._input_prompt = input_prompt or DEFAULT_SIMPLE_INPUT_PROMPT
        super().__init__(**kwargs)

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve relevant nodes."""
        del query_bundle  # Unused
        return []

    @classmethod
    def from_args(  # type: ignore
        cls,
        response_mode: ResponseMode = ResponseMode.GENERATION,
        **kwargs: Any,
    ) -> BaseGPTIndexQuery:
        if response_mode != ResponseMode.GENERATION:
            raise ValueError("response_mode should not be specified for empty query")

        return super().from_args(
            response_mode=response_mode,
            **kwargs,
        )
