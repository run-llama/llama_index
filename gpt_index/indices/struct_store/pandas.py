"""Pandas csv structured store."""

from typing import Any, Dict, Optional, Sequence, Type

import pandas as pd

from gpt_index.data_structs.table import PandasStructTable
from gpt_index.indices.base import DOCUMENTS_INPUT
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.struct_store.pandas import GPTNLPandasIndexQuery
from gpt_index.indices.struct_store.base import BaseGPTStructStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.schema import BaseDocument


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
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        df: Optional[pd.DataFrame] = None,
        index_struct: Optional[PandasStructTable] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if documents is not None:
            raise ValueError("We currently do not support indexing documents.")
        self.df = df

        super().__init__(
            documents=[],
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> PandasStructTable:
        """Build index from documents."""
        index_struct = self.index_struct_cls()
        return index_struct

    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
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
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTNLPandasIndexQuery,
        }
