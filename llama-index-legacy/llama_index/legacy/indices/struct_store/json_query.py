import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from llama_index.legacy.core.base_query_engine import BaseQueryEngine
from llama_index.legacy.core.response.schema import Response
from llama_index.legacy.prompts import BasePromptTemplate, PromptTemplate
from llama_index.legacy.prompts.default_prompts import DEFAULT_JSON_PATH_PROMPT
from llama_index.legacy.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.legacy.prompts.prompt_type import PromptType
from llama_index.legacy.schema import QueryBundle
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.utils import print_text

logger = logging.getLogger(__name__)
IMPORT_ERROR_MSG = (
    "`jsonpath_ng` package not found, please run `pip install jsonpath-ng`"
)

JSONType = Union[Dict[str, "JSONType"], List["JSONType"], str, int, float, bool, None]


DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given a query, synthesize a response "
    "to satisfy the query using the JSON results. "
    "Only include details that are relevant to the query. "
    "If you don't know the answer, then say that.\n"
    "JSON Schema: {json_schema}\n"
    "JSON Path: {json_path}\n"
    "Value at path: {json_path_value}\n"
    "Query: {query_str}\n"
    "Response: "
)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
    prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS,
)


def default_output_processor(llm_output: str, json_value: JSONType) -> JSONType:
    """Default output processor that extracts values based on JSON Path expressions."""
    # Split the given string into separate JSON Path expressions
    expressions = [expr.strip() for expr in llm_output.split(",")]

    try:
        from jsonpath_ng.ext import parse
        from jsonpath_ng.jsonpath import DatumInContext
    except ImportError as exc:
        IMPORT_ERROR_MSG = "You need to install jsonpath-ng to use this function!"
        raise ImportError(IMPORT_ERROR_MSG) from exc

    results = {}

    for expression in expressions:
        try:
            datum: List[DatumInContext] = parse(expression).find(json_value)
            if datum:
                key = expression.split(".")[
                    -1
                ]  # Extracting "title" from "$.title", for example
                results[key] = datum[0].value
        except Exception as exc:
            raise ValueError(f"Invalid JSON Path: {expression}") from exc

    return results


class JSONQueryEngine(BaseQueryEngine):
    """GPT JSON Query Engine.

    Converts natural language to JSON Path queries.

    Args:
        json_value (JSONType): JSON value
        json_schema (JSONType): JSON schema
        service_context (ServiceContext): ServiceContext
        json_path_prompt (BasePromptTemplate): The JSON Path prompt to use.
        output_processor (Callable): The output processor that executes the
            JSON Path query.
        output_kwargs (dict): Additional output processor kwargs for the
            output_processor function.
        verbose (bool): Whether to print verbose output.
    """

    def __init__(
        self,
        json_value: JSONType,
        json_schema: JSONType,
        service_context: ServiceContext,
        json_path_prompt: Optional[BasePromptTemplate] = None,
        output_processor: Optional[Callable] = None,
        output_kwargs: Optional[dict] = None,
        synthesize_response: bool = True,
        response_synthesis_prompt: Optional[BasePromptTemplate] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._json_value = json_value
        self._json_schema = json_schema
        self._service_context = service_context
        self._json_path_prompt = json_path_prompt or DEFAULT_JSON_PATH_PROMPT
        self._output_processor = output_processor or default_output_processor
        self._output_kwargs = output_kwargs or {}
        self._verbose = verbose
        self._synthesize_response = synthesize_response
        self._response_synthesis_prompt = (
            response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )

        super().__init__(self._service_context.callback_manager)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "json_path_prompt": self._json_path_prompt,
            "response_synthesis_prompt": self._response_synthesis_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "json_path_prompt" in prompts:
            self._json_path_prompt = prompts["json_path_prompt"]
        if "response_synthesis_prompt" in prompts:
            self._response_synthesis_prompt = prompts["response_synthesis_prompt"]

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def _get_schema_context(self) -> str:
        """Get JSON schema context."""
        return json.dumps(self._json_schema)

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        schema = self._get_schema_context()

        json_path_response_str = self._service_context.llm.predict(
            self._json_path_prompt,
            schema=schema,
            query_str=query_bundle.query_str,
        )

        if self._verbose:
            print_text(
                f"> JSONPath Instructions:\n" f"```\n{json_path_response_str}\n```\n"
            )

        json_path_output = self._output_processor(
            json_path_response_str,
            self._json_value,
            **self._output_kwargs,
        )

        if self._verbose:
            print_text(f"> JSONPath Output: {json_path_output}\n")

        if self._synthesize_response:
            response_str = self._service_context.llm.predict(
                self._response_synthesis_prompt,
                query_str=query_bundle.query_str,
                json_schema=self._json_schema,
                json_path=json_path_response_str,
                json_path_value=json_path_output,
            )
        else:
            response_str = json.dumps(json_path_output)

        response_metadata = {
            "json_path_response_str": json_path_response_str,
        }

        return Response(response=response_str, metadata=response_metadata)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        schema = self._get_schema_context()

        json_path_response_str = await self._service_context.llm.apredict(
            self._json_path_prompt,
            schema=schema,
            query_str=query_bundle.query_str,
        )

        if self._verbose:
            print_text(
                f"> JSONPath Instructions:\n" f"```\n{json_path_response_str}\n```\n"
            )

        json_path_output = self._output_processor(
            json_path_response_str,
            self._json_value,
            **self._output_kwargs,
        )

        if self._verbose:
            print_text(f"> JSONPath Output: {json_path_output}\n")

        if self._synthesize_response:
            response_str = await self._service_context.llm.apredict(
                self._response_synthesis_prompt,
                query_str=query_bundle.query_str,
                json_schema=self._json_schema,
                json_path=json_path_response_str,
                json_path_value=json_path_output,
            )
        else:
            response_str = json.dumps(json_path_output)

        response_metadata = {
            "json_path_response_str": json_path_response_str,
        }

        return Response(response=response_str, metadata=response_metadata)
