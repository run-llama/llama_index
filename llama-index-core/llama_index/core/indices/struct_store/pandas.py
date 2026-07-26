"""
Pandas csv structured store.

DEPRECATED AND REMOVED: This class has been removed due to security concerns
in the related PandasQueryEngine. See PandasQueryEngine module for details.
"""

from typing import Any


class PandasIndex:
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        raise DeprecationWarning(
            "PandasIndex has been removed along with PandasQueryEngine due to\n"
            "inherent security vulnerabilities in the sandboxed execution model.\n\n"
            "See llama_index.core.query_engine.pandas.PandasQueryEngine for details\n"
            "and suggested alternatives."
        )


# Legacy
GPTPandasIndex = PandasIndex
