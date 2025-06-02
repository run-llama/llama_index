"""
Pandas output parser.

DEPRECATED: This class has been moved to `llama-index-experimental`.
"""

from typing import Any


class PandasInstructionParser:
    """
    Pandas instruction parser.

    DEPRECATED: This class has been moved to `llama-index-experimental`.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise DeprecationWarning(
            "PandasInstructionParser has been moved to `llama-index-experimental`.\n"
            "`pip install llama-index-experimental`\n"
            "`from llama_index.experimental.query_engine.pandas import PandasInstructionParser`\n"
            "Note that the PandasInstructionParser allows for arbitrary code execution, \n"
            "and should be used in a secure environment."
        )
