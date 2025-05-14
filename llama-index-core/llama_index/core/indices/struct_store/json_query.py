import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_JSON_PATH_PROMPT
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.schema import QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.utils import print_text

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


def default_output_response_parser(llm_output: str) -> str:
    """Attempts to parse the JSON path prompt output. Only applicable if the default prompt is used."""
    try:
        llm_output_parsed = re.search(  # type: ignore
            pattern=r"JSONPath:\s+(.*)", string=llm_output
        ).groups()[0]
    except Exception:
        logger.warning(
            f"JSON Path could not be parsed in the LLM response after the 'JSONPath' identifier. "
            "Try passing a custom JSON path prompt and processor. "
            "Proceeding with output as-is."
        )
        return llm_output

    return llm_output_parsed


def default_output_processor(llm_output: str, json_value: JSONType) -> Dict[str, str]:
    """Default output processor that extracts values based on JSON Path expressions."""
    # Post-process the LLM output to remove the JSONPath: prefix
    llm_output = llm_output.replace("JSONPath: ", "").replace("JSON Path: ", "").strip()

    # Split the given string into separate JSON Path expressions
    expressions = [expr.strip() for expr in llm_output.split(",")]

    try:
        from jsonpath_ng.ext import parse  # pants: no-infer-dep
        from jsonpath_ng.jsonpath import DatumInContext  # pants: no-infer-dep
    except ImportError as exc:
        IMPORT_ERROR_MSG = "You need to install jsonpath-ng to use this function!"
        raise ImportError(IMPORT_ERROR_MSG) from exc

    results: Dict[str, str] = {}

    for expression in expressions:
        try:
            datum: List[DatumInContext] = parse(expression).find(json_value)
            if datum:
                key = expression.split(".")[
                    -1
                ]  # Extracting "title" from "$.title", for example
                results[key] = ", ".join(str(i.value) for i in datum)
        except Exception as exc:
            raise ValueError(f"Invalid JSON Path: {expression}") from exc

    return results


class JSONQueryEngine(BaseQueryEngine):
    """
    GPT JSON Query Engine.

    Converts natural language to JSON Path queries.

    Args:
        json_value (JSONType): JSON value
        json_schema (JSONType): JSON schema
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
        llm: Optional[LLM] = None,
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
        self._llm = llm or Settings.llm
        self._json_path_prompt = json_path_prompt or DEFAULT_JSON_PATH_PROMPT
        self._output_processor = output_processor or default_output_processor
        self._output_kwargs = output_kwargs or {}
        self._verbose = verbose
        self._synthesize_response = synthesize_response
        self._response_synthesis_prompt = (
            response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )

        super().__init__(callback_manager=Settings.callback_manager)

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

        json_path_response_str = self._llm.predict(
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

        # removes JSONPath: prefix from returned JSON path prompt call
        if self._json_path_prompt == DEFAULT_JSON_PATH_PROMPT:
            json_path_response_str = default_output_response_parser(
                json_path_response_str
            )

        if self._verbose:
            print_text(f"> JSONPath Output: {json_path_output}\n")

        if self._synthesize_response:
            response_str = self._llm.predict(
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

        json_path_response_str = await self._llm.apredict(
            self._json_path_prompt,
            schema=schema,
            query_str=query_bundle.query_str,
        )

        # removes JSONPath: prefix from returned JSON path prompt call
        if self._json_path_prompt == DEFAULT_JSON_PATH_PROMPT:
            json_path_response_str = default_output_response_parser(
                json_path_response_str
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
            response_str = await self._llm.apredict(
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
