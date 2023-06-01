import logging
from typing import Any, Optional, Dict, Callable
import json

from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.struct_store.json import GPTJSONIndex, JSONType
from llama_index.prompts.default_prompts import DEFAULT_JSON_PATH_PROMPT
from llama_index.prompts.prompts import JSONPathPrompt
from llama_index.response.schema import Response


logger = logging.getLogger(__name__)


def default_output_processor(llm_output, json_value: JSONType) -> JSONType:
    from jsonpath import JSONPath

    JSONPath(llm_output).parse(json_value)


class GPTNLJSONQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        index: GPTJSONIndex,
        json_path_prompt: Optional[JSONPathPrompt] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        output_processor: Optional[Callable] = None,
        output_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self.json_value = index.json_value
        self.json_schema = json_schema
        self._service_context = index.service_context
        self._json_path_prompt = json_path_prompt or DEFAULT_JSON_PATH_PROMPT
        self._output_processor = output_processor or default_output_processor
        self._output_kwargs = output_kwargs or {}

        super().__init__(self._service_context.callback_manager)

    def _get_table_context(self) -> str:
        """Get table context."""
        return json.dumps(self.json_schema)

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        context = self._get_table_context()

        (
            json_path_response_str,
            formatted_prompt,
        ) = self._service_context.llm_predictor.predict(
            self._json_path_prompt,
            schema=context,
            query_str=query_bundle.query_str,
        )

        json_path_output = self._output_processor(
            json_path_response_str,
            self.json_value,
            **self._output_kwargs,
        )

        response_extra_info = {
            "json_path_response_str": json_path_response_str,
        }

        return Response(response=json_path_output, extra_info=response_extra_info)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)
