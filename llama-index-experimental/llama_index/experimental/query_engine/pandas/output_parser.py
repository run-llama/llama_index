"""Pandas output parser."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from llama_index.experimental.exec_utils import safe_eval, safe_exec
from llama_index.core.output_parsers.base import ChainableOutputParser
from llama_index.core.output_parsers.utils import parse_code_markdown

logger = logging.getLogger(__name__)


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

    local_vars = {"df": df, "pd": pd}
    global_vars = {"np": np}

    output = parse_code_markdown(output, only_last=True)[0]

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
            module_end_str = safe_eval(module_end_str, global_vars, local_vars)
        try:
            # str(pd.dataframe) will truncate output by display.max_colwidth
            # set width temporarily to extract more text
            current_max_colwidth = pd.get_option("display.max_colwidth")
            current_max_rows = pd.get_option("display.max_rows")
            current_max_columns = pd.get_option("display.max_columns")
            if "max_colwidth" in output_kwargs:
                pd.set_option("display.max_colwidth", output_kwargs["max_colwidth"])
            if "max_rows" in output_kwargs:
                pd.set_option("display.max_rows", output_kwargs["max_rows"])
            if "max_columns" in output_kwargs:
                pd.set_option("display.max_columns", output_kwargs["max_columns"])
            output_str = str(safe_eval(module_end_str, global_vars, local_vars))
            pd.set_option("display.max_colwidth", current_max_colwidth)
            pd.set_option("display.max_rows", current_max_rows)
            pd.set_option("display.max_columns", current_max_columns)
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


class PandasInstructionParser(ChainableOutputParser):
    """Pandas instruction parser.

    This 'output parser' takes in pandas instructions (in Python code) and
    executes them to return an output.

    """

    def __init__(
        self, df: pd.DataFrame, output_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize params."""
        self.df = df
        self.output_kwargs = output_kwargs or {}

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        return default_output_processor(output, self.df, **self.output_kwargs)
