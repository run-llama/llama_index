"""Query Schema.

This schema is used under the hood for all queries, but is primarily
exposed for recursive queries over composable indices.

"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from dataclasses_json import DataClassJsonMixin


class QueryMode(str, Enum):
    """Query mode enum.

    Can be passed as the enum struct, or as the underlying string.

    Attributes:
        DEFAULT ("default"): Default query mode.
        RETRIEVE ("retrieve"): Retrieve mode.
        EMBEDDING ("embedding"): Embedding mode.
        SUMMARIZE ("summarize"): Summarize mode. Used for hierarchical
            summarization in the tree index.
        SIMPLE ("simple"): Simple mode. Used for keyword extraction.
        RAKE ("rake"): RAKE mode. Used for keyword extraction.
        RECURSIVE ("recursive"): Recursive mode. Used to recursively query
            over composed indices.

    """

    DEFAULT = "default"
    # a special "retrieve" query for tree index that retrieves that top nodes
    RETRIEVE = "retrieve"
    # embedding-based query
    EMBEDDING = "embedding"

    # to hierarchically summarize using tree
    SUMMARIZE = "summarize"

    # for keyword extractor
    SIMPLE = "simple"
    RAKE = "rake"

    # recursive queries (composable queries)
    # NOTE: deprecated
    RECURSIVE = "recursive"

    # for sql queries
    SQL = "sql"


@dataclass
class QueryBundle(DataClassJsonMixin):
    """
    Query bundle.

    This dataclass contains the original query string and associated transformations.

    Args:
        query_str (str): the original user-specified query string.
            This is currently used by all non embedding-based queries.
        embedding_strs (list[str]): list of strings used for embedding the query.
            This is currently used by all embedding-based queries.
        embedding (list[float]): the stored embedding for the query.
    """

    query_str: str
    custom_embedding_strs: Optional[List[str]] = None
    embedding: Optional[List[float]] = None

    @property
    def embedding_strs(self) -> List[str]:
        """Use custom embedding strs if specified, otherwise use query str."""
        if self.custom_embedding_strs is None:
            return [self.query_str]
        else:
            return self.custom_embedding_strs
