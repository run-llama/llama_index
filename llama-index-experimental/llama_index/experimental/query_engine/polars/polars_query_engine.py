"""
Default query for PolarsIndex.

WARNING: This tool provides the LLM with access to the `eval` function.
Arbitrary code execution is possible on the machine running this tool.
This tool is not recommended to be used in a production setting, and would
require heavy sandboxing or virtual machines

"""

import logging
from typing import Any, Dict, Optional

import polars as pl
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.schema import QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.utils import print_text
from llama_index.experimental.query_engine.polars.prompts import DEFAULT_POLARS_PROMPT
from llama_index.experimental.query_engine.polars.output_parser import (
    PolarsInstructionParser,
)

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTION_STR = (
    "1. Convert the query to executable Python code using Polars.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)


# **NOTE**: newer version of sql query engine
DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Polars Instructions (optional):\n{polars_instructions}\n\n"
    "Polars Output: {polars_output}\n\n"
    "Response: "
)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
)


class PolarsQueryEngine(BaseQueryEngine):
    """
    Polars query engine.

    Convert natural language to Polars python code.

    WARNING: This tool provides the Agent access to the `eval` function.
    Arbitrary code execution is possible on the machine running this tool.
    This tool is not recommended to be used in a production setting, and would
    require heavy sandboxing or virtual machines


    Args:
        df (pl.DataFrame): Polars dataframe to use.
        instruction_str (Optional[str]): Instruction string to use.
        instruction_parser (Optional[PolarsInstructionParser]): The output parser
            that takes the polars query output string and returns a string.
            It defaults to PolarsInstructionParser and takes polars DataFrame,
            and any output kwargs as parameters.
        polars_prompt (Optional[BasePromptTemplate]): Polars prompt to use.
        output_kwargs (dict): Additional output processor kwargs for the
            PolarsInstructionParser.
        head (int): Number of rows to show in the table context.
        verbose (bool): Whether to print verbose output.
        llm (Optional[LLM]): Language model to use.
        synthesize_response (bool): Whether to synthesize a response from the
            query results. Defaults to False.
        response_synthesis_prompt (Optional[BasePromptTemplate]): A
            Response Synthesis BasePromptTemplate to use for the query. Defaults to
            DEFAULT_RESPONSE_SYNTHESIS_PROMPT.

    Examples:
        `pip install llama-index-experimental polars`

        ```python
        import polars as pl
        from llama_index.experimental.query_engine.polars import PolarsQueryEngine

        df = pl.DataFrame(
            {
                "city": ["Toronto", "Tokyo", "Berlin"],
                "population": [2930000, 13960000, 3645000]
            }
        )

        query_engine = PolarsQueryEngine(df=df, verbose=True)

        response = query_engine.query("What is the population of Tokyo?")
        ```

    """

    def __init__(
        self,
        df: pl.DataFrame,
        instruction_str: Optional[str] = None,
        instruction_parser: Optional[PolarsInstructionParser] = None,
        polars_prompt: Optional[BasePromptTemplate] = None,
        output_kwargs: Optional[dict] = None,
        head: int = 5,
        verbose: bool = False,
        llm: Optional[LLM] = None,
        synthesize_response: bool = False,
        response_synthesis_prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._df = df

        self._head = head
        self._polars_prompt = polars_prompt or DEFAULT_POLARS_PROMPT
        self._instruction_str = instruction_str or DEFAULT_INSTRUCTION_STR
        self._instruction_parser = instruction_parser or PolarsInstructionParser(
            df, output_kwargs or {}
        )
        self._verbose = verbose

        self._llm = llm or Settings.llm
        self._synthesize_response = synthesize_response
        self._response_synthesis_prompt = (
            response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )

        super().__init__(callback_manager=Settings.callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "polars_prompt": self._polars_prompt,
            "response_synthesis_prompt": self._response_synthesis_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "polars_prompt" in prompts:
            self._polars_prompt = prompts["polars_prompt"]
        if "response_synthesis_prompt" in prompts:
            self._response_synthesis_prompt = prompts["response_synthesis_prompt"]

    def _get_table_context(self) -> str:
        """Get table context."""
        return str(self._df.head(self._head))

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        context = self._get_table_context()

        polars_response_str = self._llm.predict(
            self._polars_prompt,
            df_str=context,
            query_str=query_bundle.query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            print_text(f"> Polars Instructions:\n```\n{polars_response_str}\n```\n")
        polars_output = self._instruction_parser.parse(polars_response_str)
        if self._verbose:
            print_text(f"> Polars Output: {polars_output}\n")

        response_metadata = {
            "polars_instruction_str": polars_response_str,
            "raw_polars_output": polars_output,
        }
        if self._synthesize_response:
            response_str = str(
                self._llm.predict(
                    self._response_synthesis_prompt,
                    query_str=query_bundle.query_str,
                    polars_instructions=polars_response_str,
                    polars_output=polars_output,
                )
            )
        else:
            response_str = str(polars_output)

        return Response(response=response_str, metadata=response_metadata)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Answer a query asynchronously."""
        context = self._get_table_context()

        polars_response_str = await self._llm.apredict(
            self._polars_prompt,
            df_str=context,
            query_str=query_bundle.query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            print_text(f"> Polars Instructions:\n```\n{polars_response_str}\n```\n")
        polars_output = self._instruction_parser.parse(polars_response_str)
        if self._verbose:
            print_text(f"> Polars Output: {polars_output}\n")

        response_metadata = {
            "polars_instruction_str": polars_response_str,
            "raw_polars_output": polars_output,
        }
        if self._synthesize_response:
            response_str = str(
                await self._llm.apredict(
                    self._response_synthesis_prompt,
                    query_str=query_bundle.query_str,
                    polars_instructions=polars_response_str,
                    polars_output=polars_output,
                )
            )
        else:
            response_str = str(polars_output)

        return Response(response=response_str, metadata=response_metadata)


# legacy
NLPolarsQueryEngine = PolarsQueryEngine
GPTNLPolarsQueryEngine = PolarsQueryEngine
