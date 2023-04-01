"""Pandas csv structured store."""

from typing import Any, Optional, Sequence

from gpt_index.data_structs.node_v2 import Node
from gpt_index.data_structs.table_v2 import PandasStructTable
from gpt_index.indices.base import QueryMap
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.struct_store.base import BaseGPTStructStoreIndex
from gpt_index.indices.struct_store.pandas_query import GPTNLPandasIndexQuery

import pandas as pd


class GPTPandasIndex(BaseGPTStructStoreIndex[PandasStructTable]):
    """Base GPT Pandas Index.

    The GPTPandasStructStoreIndex is an index that stores
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
        nodes: Optional[Sequence[Node]] = None,
        df: Optional[pd.DataFrame] = None,
        index_struct: Optional[PandasStructTable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if nodes is not None:
            raise ValueError("We currently do not support indexing documents or nodes.")
        self.df = df

        super().__init__(
            nodes=[],
            index_struct=index_struct,
            **kwargs,
        )

    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> PandasStructTable:
        """Build index from documents."""
        index_struct = self.index_struct_cls()
        return index_struct

    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert a document."""
        raise NotImplementedError("We currently do not support inserting documents.")

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query.

        This allows subclasses to pass in additional query kwargs
        to query, for instance arguments that are shared between the
        index and the query class. By default, this does nothing.
        This also allows subclasses to do validation.

        """
        super()._preprocess_query(mode, query_kwargs)
        # pass along sql_database, table_name
        query_kwargs["df"] = self.df

    @classmethod
    def get_query_map(self) -> QueryMap:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTNLPandasIndexQuery,
        }
