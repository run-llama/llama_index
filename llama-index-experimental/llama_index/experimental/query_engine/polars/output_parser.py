"""Polars output parser."""

import ast
import logging
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from llama_index.core.output_parsers import BaseOutputParser
from llama_index.core.output_parsers.utils import parse_code_markdown
from llama_index.experimental.exec_utils import safe_eval, safe_exec

logger = logging.getLogger(__name__)


def default_output_processor(
    output: str, df: pl.DataFrame, **output_kwargs: Any
) -> str:
    """Process outputs in a default manner."""
    if sys.version_info < (3, 9):
        logger.warning(
            "Python version must be >= 3.9 in order to use "
            "the default output processor, which executes "
            "the Python query. Instead, we will return the "
            "raw Python instructions as a string."
        )
        return output

    local_vars = {"df": df, "pl": pl}
    global_vars = {"np": np}

    output = parse_code_markdown(output, only_last=True)
    if not isinstance(output, str):
        output = output[0]

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
            # Handle Polars DataFrame display options
            result = safe_eval(module_end_str, global_vars, local_vars)

            # Set display options for Polars if provided
            if isinstance(result, pl.DataFrame):
                # Polars doesn't have global display options like pandas,
                # but we can control the output format
                if "max_rows" in output_kwargs:
                    max_rows = output_kwargs["max_rows"]
                    if len(result) > max_rows:
                        # Show head and tail with indication of truncation
                        head_rows = max_rows // 2
                        tail_rows = max_rows - head_rows
                        result_str = (
                            str(result.head(head_rows))
                            + "\n...\n"
                            + str(result.tail(tail_rows))
                        )
                    else:
                        result_str = str(result)
                else:
                    result_str = str(result)
            else:
                result_str = str(result)

            return result_str

        except Exception:
            raise
    except Exception as e:
        err_string = (
            f"There was an error running the output as Python code. Error message: {e}"
        )
        traceback.print_exc()
        return err_string


class PolarsInstructionParser(BaseOutputParser):
    """
    Polars instruction parser.

    This 'output parser' takes in polars instructions (in Python code) and
    executes them to return an output.

    """

    def __init__(
        self, df: pl.DataFrame, output_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize params."""
        self.df = df
        self.output_kwargs = output_kwargs or {}

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        return default_output_processor(output, self.df, **self.output_kwargs)
