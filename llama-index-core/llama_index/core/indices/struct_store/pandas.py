"""Pandas csv structured store.

DEPRECATED: Please use :class:`PandasQueryEngine` in `llama-index-experimental` instead.
"""

from typing import Any


class PandasIndex:
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        raise DeprecationWarning(
            "PandasQueryEngine has been moved to `llama-index-experimental`.\n"
            "`pip install llama-index-experimental`\n"
            "`from llama_index.experimental.query_engine import PandasQueryEngine`"
        )


# Legacy
GPTPandasIndex = PandasIndex
