"""Pandas csv structured store."""

import logging
from typing import Any, Optional, Sequence

import pandas as pd
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.data_structs.table import PandasStructTable
from llama_index.core.indices.struct_store.base import BaseStructStoreIndex
from llama_index.core.llms.utils import LLMType
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)


class PandasIndex(BaseStructStoreIndex[PandasStructTable]):
    """Pandas Index.

    Deprecated. Please use :class:`PandasQueryEngine` instead.

    The PandasIndex is an index that stores
    a Pandas dataframe under the hood.
    Currently index "construction" is not supported.

    During query time, the user can either specify a raw SQL query
    or a natural language query to retrieve their data.

    Args:
        pandas_df (Optional[pd.DataFrame]): Pandas dataframe to use.
            See :ref:`Ref-Struct-Store` for more details.

    """

    index_struct_cls = PandasStructTable

    def __init__(
        self,
        df: pd.DataFrame,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[PandasStructTable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        logger.warning(
            "PandasIndex is deprecated. \
            Please directly use `PandasQueryEngine(df)` instead."
        )

        if nodes is not None:
            raise ValueError("We currently do not support indexing documents or nodes.")
        self.df = df

        super().__init__(
            nodes=[],
            index_struct=index_struct,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        raise NotImplementedError("Not supported")

    def as_query_engine(
        self, llm: Optional[LLMType] = None, **kwargs: Any
    ) -> BaseQueryEngine:
        # NOTE: lazy import
        from llama_index.core.query_engine.pandas.pandas_query_engine import (
            PandasQueryEngine,
        )

        return PandasQueryEngine.from_index(self, llm=llm, **kwargs)

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> PandasStructTable:
        """Build index from documents."""
        return self.index_struct_cls()

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        raise NotImplementedError("We currently do not support inserting documents.")


# legacy
GPTPandasIndex = PandasIndex
