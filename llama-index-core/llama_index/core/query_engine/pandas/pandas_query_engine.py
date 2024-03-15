"""Default query for PandasIndex.

WARNING: This tool provides the Agent access to the `eval` function.
Arbitrary code execution is possible on the machine running this tool.
This tool is not recommended to be used in a production setting, and would
require heavy sandboxing or virtual machines

"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.indices.struct_store.pandas import PandasIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_PANDAS_PROMPT
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.query_engine.pandas.output_parser import (
    PandasInstructionParser,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.utils import print_text

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTION_STR = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)


# **NOTE**: newer version of sql query engine
DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
)


class PandasQueryEngine(BaseQueryEngine):
    """Pandas query engine.

    Convert natural language to Pandas python code.

    WARNING: This tool provides the Agent access to the `eval` function.
    Arbitrary code execution is possible on the machine running this tool.
    This tool is not recommended to be used in a production setting, and would
    require heavy sandboxing or virtual machines


    Args:
        df (pd.DataFrame): Pandas dataframe to use.
        instruction_str (Optional[str]): Instruction string to use.
        output_processor (Optional[Callable[[str], str]]): Output processor.
            A callable that takes in the output string, pandas DataFrame,
            and any output kwargs and returns a string.
            eg.kwargs["max_colwidth"] = [int] is used to set the length of text
            that each column can display during str(df). Set it to a higher number
            if there is possibly long text in the dataframe.
        pandas_prompt (Optional[BasePromptTemplate]): Pandas prompt to use.
        head (int): Number of rows to show in the table context.
        llm (Optional[LLM]): Language model to use.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        instruction_str: Optional[str] = None,
        instruction_parser: Optional[PandasInstructionParser] = None,
        pandas_prompt: Optional[BasePromptTemplate] = None,
        output_kwargs: Optional[dict] = None,
        head: int = 5,
        verbose: bool = False,
        service_context: Optional[ServiceContext] = None,
        llm: Optional[LLM] = None,
        synthesize_response: bool = False,
        response_synthesis_prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._df = df

        self._head = head
        self._pandas_prompt = pandas_prompt or DEFAULT_PANDAS_PROMPT
        self._instruction_str = instruction_str or DEFAULT_INSTRUCTION_STR
        self._instruction_parser = instruction_parser or PandasInstructionParser(
            df, output_kwargs or {}
        )
        self._verbose = verbose

        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._synthesize_response = synthesize_response
        self._response_synthesis_prompt = (
            response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )

        super().__init__(
            callback_manager=callback_manager_from_settings_or_context(
                Settings, service_context
            )
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "pandas_prompt": self._pandas_prompt,
            "response_synthesis_prompt": self._response_synthesis_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "pandas_prompt" in prompts:
            self._pandas_prompt = prompts["pandas_prompt"]
        if "response_synthesis_prompt" in prompts:
            self._response_synthesis_prompt = prompts["response_synthesis_prompt"]

    @classmethod
    def from_index(cls, index: PandasIndex, **kwargs: Any) -> "PandasQueryEngine":
        logger.warning(
            "PandasIndex is deprecated. "
            "Directly construct PandasQueryEngine with df instead."
        )
        return cls(df=index.df, service_context=index.service_context, **kwargs)

    def _get_table_context(self) -> str:
        """Get table context."""
        return str(self._df.head(self._head))

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        context = self._get_table_context()

        pandas_response_str = self._llm.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_bundle.query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            print_text(f"> Pandas Instructions:\n" f"```\n{pandas_response_str}\n```\n")
        pandas_output = self._instruction_parser.parse(pandas_response_str)
        if self._verbose:
            print_text(f"> Pandas Output: {pandas_output}\n")

        response_metadata = {
            "pandas_instruction_str": pandas_response_str,
            "raw_pandas_output": pandas_output,
        }
        if self._synthesize_response:
            response_str = str(
                self._llm.predict(
                    self._response_synthesis_prompt,
                    query_str=query_bundle.query_str,
                    pandas_instructions=pandas_response_str,
                    pandas_output=pandas_output,
                )
            )
        else:
            response_str = str(pandas_output)

        return Response(response=response_str, metadata=response_metadata)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)


# legacy
NLPandasQueryEngine = PandasQueryEngine
GPTNLPandasQueryEngine = PandasQueryEngine
