import logging
from typing import Any, Optional, Dict, Callable, List
import json
from langchain.input import print_text

from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.struct_store.json import GPTJSONIndex, JSONType
from llama_index.prompts.default_prompts import DEFAULT_JSON_PATH_PROMPT
from llama_index.prompts.prompts import JSONPathPrompt
from llama_index.response.schema import Response


logger = logging.getLogger(__name__)


def default_output_processor(llm_output, json_value: JSONType) -> JSONType:
    from jsonpath_ng.ext import parse
    from jsonpath_ng.jsonpath import DatumInContext

    datum: List[DatumInContext] = parse(llm_output).find(json_value)
    return [d.value for d in datum]


class GPTNLJSONQueryEngine(BaseQueryEngine):
    """GPT NL JSON Query Engine.

    Converts natural language to JSON Path queries.

    Args:
        index (GPTJSONIndex): The index to query.
        json_path_prompt (Prompt): The JSON Path prompt to use.
        output_processor (Callable): The output processor that executes the JSON Path query.
        output_kwargs (dict): Additional output processor kwargs for the output_processor function.
        verbose (bool): Whether to print verbose output.
    """
    def __init__(
        self,
        index: GPTJSONIndex,
        json_path_prompt: Optional[JSONPathPrompt] = None,
        output_processor: Optional[Callable] = None,
        output_kwargs: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self.json_value = index.json_value
        self.json_schema = index.json_schema
        self._service_context = index.service_context
        self._json_path_prompt = json_path_prompt or DEFAULT_JSON_PATH_PROMPT
        self._output_processor = output_processor or default_output_processor
        self._output_kwargs = output_kwargs or {}
        self._verbose = verbose

        super().__init__(self._service_context.callback_manager)

    def _get_schema_context(self) -> str:
        """Get JSON schema context."""
        return json.dumps(self.json_schema)

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        schema = self._get_schema_context()

        (
            json_path_response_str,
            formatted_prompt,
        ) = self._service_context.llm_predictor.predict(
            self._json_path_prompt,
            schema=schema,
            query_str=query_bundle.query_str,
        )

        if self._verbose:
            print_text(f"> JSONPath Prompt: {formatted_prompt}\n")
            print_text(
                f"> JSONPath Instructions:\n" f"```\n{json_path_response_str}\n```\n"
            )

        json_path_output = self._output_processor(
            json_path_response_str,
            self.json_value,
            **self._output_kwargs,
        )

        if self._verbose:
            print_text(f"> JSONPath Output: {json_path_output}\n")

        response_extra_info = {
            "json_path_response_str": json_path_response_str,
        }

        return Response(response=json.dumps(json_path_output), extra_info=response_extra_info)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)
