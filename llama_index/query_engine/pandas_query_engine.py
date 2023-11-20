"""Default query for PandasIndex.

WARNING: This tool provides the Agent access to the `eval` function.
Arbitrary code execution is possible on the machine running this tool.
This tool is not recommended to be used in a production setting, and would
require heavy sandboxing or virtual machines

"""

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from llama_index.core import BaseQueryEngine
from llama_index.exec_utils import safe_eval, safe_exec
from llama_index.indices.struct_store.pandas import PandasIndex
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.default_prompts import DEFAULT_PANDAS_PROMPT
from llama_index.prompts.mixin import PromptMixinType
from llama_index.response.schema import Response
from llama_index.schema import QueryBundle
from llama_index.service_context import ServiceContext
from llama_index.utils import print_text

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTION_STR = (
    "We wish to convert this query to executable Python code using Pandas.\n"
    "The final line of code should be a Python expression that can be called "
    "with the `eval()` function. This expression should represent a solution "
    "to the query. This expression should not have leading or trailing "
    "quotes.\n"
)


def default_output_processor(
    output: str, df: pd.DataFrame, **output_kwargs: Any
) -> str:
    """Process outputs in a default manner."""
    import ast
    import sys
    import traceback

    if sys.version_info < (3, 9):
        logger.warning(
            "Python version must be >= 3.9 in order to use "
            "the default output processor, which executes "
            "the Python query. Instead, we will return the "
            "raw Python instructions as a string."
        )
        return output

    local_vars = {"df": df}

    # NOTE: inspired from langchain's tool
    # see langchain.tools.python.tool (PythonAstREPLTool)
    try:
        tree = ast.parse(output)
        module = ast.Module(tree.body[:-1], type_ignores=[])
        safe_exec(ast.unparse(module), {}, local_vars)  # type: ignore
        module_end = ast.Module(tree.body[-1:], type_ignores=[])
        module_end_str = ast.unparse(module_end)  # type: ignore
        if module_end_str.strip("'\"") != module_end_str:
            # if there's leading/trailing quotes, then we need to eval
            # string to get the actual expression
            module_end_str = safe_eval(module_end_str, {"np": np}, local_vars)
        try:
            # str(pd.dataframe) will truncate output by display.max_colwidth
            # set width temporarily to extract more text
            if "max_colwidth" in output_kwargs:
                pd.set_option("display.max_colwidth", output_kwargs["max_colwidth"])
            output_str = str(safe_eval(module_end_str, {"np": np}, local_vars))
            pd.reset_option("display.max_colwidth")
            return output_str

        except Exception:
            raise
    except Exception as e:
        err_string = (
            "There was an error running the output as Python code. "
            f"Error message: {e}"
        )
        traceback.print_exc()
        return err_string


class PandasQueryEngine(BaseQueryEngine):
    """GPT Pandas query.

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

    """

    def __init__(
        self,
        df: pd.DataFrame,
        instruction_str: Optional[str] = None,
        output_processor: Optional[Callable] = None,
        pandas_prompt: Optional[BasePromptTemplate] = None,
        output_kwargs: Optional[dict] = None,
        head: int = 5,
        verbose: bool = False,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._df = df

        self._head = head
        self._pandas_prompt = pandas_prompt or DEFAULT_PANDAS_PROMPT
        self._instruction_str = instruction_str or DEFAULT_INSTRUCTION_STR
        self._output_processor = output_processor or default_output_processor
        self._output_kwargs = output_kwargs or {}
        self._verbose = verbose
        self._service_context = service_context or ServiceContext.from_defaults()

        super().__init__(self._service_context.callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {"pandas_prompt": self._pandas_prompt}

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

        pandas_response_str = self._service_context.llm_predictor.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_bundle.query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            print_text(f"> Pandas Instructions:\n" f"```\n{pandas_response_str}\n```\n")
        pandas_output = self._output_processor(
            pandas_response_str,
            self._df,
            **self._output_kwargs,
        )
        if self._verbose:
            print_text(f"> Pandas Output: {pandas_output}\n")

        response_metadata = {
            "pandas_instruction_str": pandas_response_str,
        }

        return Response(response=pandas_output, metadata=response_metadata)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)


# legacy
NLPandasQueryEngine = PandasQueryEngine
GPTNLPandasQueryEngine = PandasQueryEngine
