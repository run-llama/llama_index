"""Shared type declarations for the Apache Solr vector store integration."""

from typing import TypedDict

from pydantic import BaseModel
from typing_extensions import NotRequired


class BoostedTextField(BaseModel):
    """
    A text field with an optional boost value for Solr queries.

    This model represents a Solr field that can have a multiplicative boost
    factor applied to increase or decrease its relevance in search results.
    Boost factors greater than 1.0 increase relevance, while factors between
    0.0 and 1.0 decrease it.

    Attributes:
    field: The Solr field name to include in the search.
    boost_factor: The boost multiplier to apply. Defaults
        to 1.0 (no boost). Values > 1.0 increase relevance, 0.0 < values < 1.0
        decrease it.

    """

    field: str
    boost_factor: float = 1.0

    def get_query_str(self) -> str:  # pragma: no cover
        """
        Return Solr query syntax representation for this field.

        If the boost factor is 1.0 (default) the field term is returned as-is;
        otherwise the canonical Solr boost syntax ``field^boost_factor`` is produced.
        """
        if self.boost_factor != 1.0:
            return f"{self.field}^{self.boost_factor}"
        return self.field


class SolrQueryDict(TypedDict):
    """
    Dictionary representing a Solr query with parameters.

    This is not an exhaustive list of Solr parameters, only those currently
    used by the vector store implementation.
    """

    q: str
    fq: list[str]
    fl: NotRequired[str]
    rows: NotRequired[str]
