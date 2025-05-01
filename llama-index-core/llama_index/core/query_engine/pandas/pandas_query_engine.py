"""
Default query for PandasIndex.

WARNING: This tool provides the LLM access to the `eval` function.
Arbitrary code execution is possible on the machine running this tool.
This tool is not recommended to be used in a production setting, and would
require heavy sandboxing or virtual machines.

DEPRECATED: Use `PandasQueryEngine` from `llama-index-experimental` instead.

"""

from typing import Any


class PandasQueryEngine:
    """
    Pandas query engine.

    DEPRECATED: Use `PandasQueryEngine` from `llama-index-experimental` instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise DeprecationWarning(
            "PandasQueryEngine has been moved to `llama-index-experimental`.\n"
            "`pip install llama-index-experimental`\n"
            "`from llama_index.experimental.query_engine import PandasQueryEngine`\n"
            "Note that the PandasQueryEngine allows for arbitrary code execution, \n"
            "and should be used in a secure environment."
        )


# legacy
NLPandasQueryEngine = PandasQueryEngine
GPTNLPandasQueryEngine = PandasQueryEngine
