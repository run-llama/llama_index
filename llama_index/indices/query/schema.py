"""Query Schema.

This schema is used under the hood for all queries, but is primarily
exposed for recursive queries over composable indices.

"""

from dataclasses import dataclass
from typing import List, Optional, Union

from dataclasses_json import DataClassJsonMixin


@dataclass
class QueryBundle(DataClassJsonMixin):
    """
    Query bundle.

    This dataclass contains the original query string and associated transformations.

    Args:
        query_str (str): the original user-specified query string.
            This is currently used by all non embedding-based queries.
        image_path (str): image used for embedding the image query.
        custom_embedding_strs (list[str]): list of strings used for embedding the query.
            This is currently used by all embedding-based queries.
        embedding (list[float]): the stored embedding for the query.
    """

    query_str: str
    # using single image path as query input
    image_path: Optional[str] = None
    custom_embedding_strs: Optional[List[str]] = None
    embedding: Optional[List[float]] = None

    @property
    def embedding_strs(self) -> List[str]:
        """Use custom embedding strs if specified, otherwise use query str."""
        if self.custom_embedding_strs is None:
            if len(self.query_str) == 0:
                return []
            return [self.query_str]
        else:
            return self.custom_embedding_strs

    @property
    def embedding_image(self) -> List[any]:
        """Use image path for image retrieval."""
        if self.image_path is None:
            return []
        return [self.image_path]


QueryType = Union[str, QueryBundle]
