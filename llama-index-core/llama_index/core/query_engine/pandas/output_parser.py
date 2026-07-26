"""
Pandas output parser.

DEPRECATED AND REMOVED: This class has been removed due to security concerns
in the related PandasQueryEngine. See PandasQueryEngine module for details.
"""

from typing import Any


class PandasInstructionParser:
    """
    Pandas instruction parser.

    DEPRECATED AND REMOVED: This class has been removed due to security concerns
    in the related PandasQueryEngine.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise DeprecationWarning(
            "PandasInstructionParser has been removed along with PandasQueryEngine\n"
            "due to inherent security vulnerabilities in the sandboxed execution model.\n\n"
            "See llama_index.core.query_engine.pandas.PandasQueryEngine for details\n"
            "and suggested alternatives."
        )
