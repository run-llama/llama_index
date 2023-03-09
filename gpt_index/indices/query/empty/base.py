"""Default query for GPTEmptyIndex."""
from typing import Any, Optional

from gpt_index.data_structs.data_structs import EmptyIndex
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from gpt_index.prompts.prompts import SimpleInputPrompt
from gpt_index.response.schema import RESPONSE_TYPE, Response, StreamingResponse


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
        index_struct: EmptyIndex,
        input_prompt: Optional[SimpleInputPrompt] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._input_prompt = input_prompt or DEFAULT_SIMPLE_INPUT_PROMPT
        super().__init__(index_struct=index_struct, **kwargs)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        if not self._streaming:
            response, _ = self._llm_predictor.predict(
                self._input_prompt,
                query_str=query_bundle.query_str,
            )
            return Response(response)
        else:
            stream_response, _ = self._llm_predictor.stream(
                self._input_prompt,
                query_str=query_bundle.query_str,
            )
            return StreamingResponse(stream_response)
