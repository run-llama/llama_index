"""Shared constants for use with the Solr vector store."""

from types import SimpleNamespace
from typing import Final


class SolrConstants(SimpleNamespace):
    """Constants used by Solr clients."""

    QUERY_ALL: Final[str] = "*:*"
    """Solr query requesting all documents to be returned."""

    DEFAULT_TIMEOUT_SEC: Final[int] = 60
    """Default request timeout to Solr in seconds."""

    SOLR_ISO8601_DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
    """A :py:meth:`datetime.datetime.strftime` format string for Solr-compatible datetimes.

    See `Solr documentation
    <https://solr.apache.org/guide/solr/latest/indexing-guide/date-formatting-math.html>`_
    for more information.
    """
