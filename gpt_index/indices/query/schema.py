"""Query Configuration Schema.

This schema is used under the hood for all queries, but is primarily
exposed for recursive queries over composable indices.

"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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
class QueryConfig(DataClassJsonMixin):
    """Query config.

    Used under the hood for all queries.
    The user must explicitly specify a list of query config objects is passed during
    a query call to define configurations for each individual subindex within an
    overall composed index.

    The user may choose to specify either the query config objects directly,
    or as a list of JSON dictionaries. For instance, the following are equivalent:

    .. code-block:: python

        # using JSON dictionaries
        query_configs = [
            {
                # index_struct_id is optional
                "index_struct_id": "<index_struct_id>",
                "index_struct_type": "tree",
                "query_mode": "default",
                "query_kwargs": {
                    "child_branch_factor": 2
                }
            },
            ...
        ]
        response = index.query(
            "<query_str>", mode="recursive", query_configs=query_configs
        )

    .. code-block:: python

        query_configs = [
            QueryConfig(
                index_struct_id="<index_struct_id>",
                index_struct_type=IndexStructType.TREE,
                query_mode=QueryMode.DEFAULT,
                query_kwargs={
                    "child_branch_factor": 2
                }
            )
            ...
        ]
        response = index.query(
            "<query_str>", mode="recursive", query_configs=query_configs
        )


    Args:
        index_struct_id (Optional[str]): The index struct id. This can be obtained
            by calling
            "get_doc_id" on the original index class. This can be set by calling
            "set_doc_id" on the original index class.
        index_struct_type (IndexStructType): The type of index struct.
        query_mode (QueryMode): The query mode.
        query_kwargs (Dict[str, Any], optional): The query kwargs. Defaults to {}.

    """

    # index_struct_type: IndexStructType
    index_struct_type: str
    query_mode: QueryMode
    query_kwargs: Dict[str, Any] = field(default_factory=dict)
    # NOTE: type as Optional because old query configs may not
    # have this field
    index_struct_id: Optional[str] = None


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
    """

    query_str: str
    custom_embedding_strs: Optional[List[str]] = None

    @property
    def embedding_strs(self) -> List[str]:
        """Use custom embedding strs if specified, otherwise use query str."""
        if self.custom_embedding_strs is None:
            return [self.query_str]
        else:
            return self.custom_embedding_strs
