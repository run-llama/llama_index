"""
Constants for Solr vector store.
This module contains configuration constants, escape rules, and field definitions
used throughout the Solr vector store implementation. These constants ensure
consistent behavior across different components and provide centralized
configuration for Solr-specific operations.
The constants are organized into the following categories:
- Tokenization and delimiter constants
- Sparse encoding field definitions
- Query escaping rules for different Solr query parsers
- Default configuration values
"""

from types import MappingProxyType, SimpleNamespace
from typing import Final

# =============================================================================
# Configuration Defaults
# =============================================================================

SOLR_DEFAULT_MINIMUM_CHUNK_SIZE: Final[int] = 50
"""Default minimum size (in characters) for text chunks during document processing."""

# =============================================================================
# Query Escaping Rules
# =============================================================================

# Private mapping of characters that need escaping in Solr queries
_ESCAPE_RULES_MAPPING = {
    "/": r"\/",
    "'": r"\'",
    "\\": r"\\\\",
    "+": r"\+",
    "-": r"\-",
    "&": r"\&",
    "|": r"\|",
    "!": r"\!",
    "(": r"\(",
    ")": r"\)",
    "{": r"\{",
    "}": r"\}",
    "[": r"\[",
    "]": r"\]",
    "^": r"\^",
    "~": r"\~",
    "*": r"\*",
    "?": r"\?",
    ":": r"\:",
    '"': r"\"",
    ";": r"\;",
    " ": r"\ ",
}


ESCAPE_RULES_GENERIC = MappingProxyType[int, str](str.maketrans(_ESCAPE_RULES_MAPPING))
"""Translation table for escaping special characters in standard Solr queries.
This mapping is used with str.translate() to escape characters that have special
meaning in Solr's query syntax.
Example:
    escaped_query = user_input.translate(ESCAPE_RULES_GENERIC)
"""

ESCAPE_RULES_NESTED_LUCENE_DISMAX = MappingProxyType[int, str](
    str.maketrans(
        {
            **_ESCAPE_RULES_MAPPING,
            "+": r"\\+",  # Double-escaped plus
            "-": r"\\-",  # Double-escaped minus
        }
    )
)
"""Translation table for escaping characters in nested Lucene+DisMax queries.
Double escaping for dismax special characters. Since we have two nested query parsers
(``lucene`` + ``dismax``), Solr parses the escaping characters twice, requiring
additional escaping for certain operators.
Use this when constructing queries that will be processed by both the Lucene
query parser and the DisMax query parser in sequence.
Example:
    escaped_query = user_input.translate(ESCAPE_RULES_NESTED_LUCENE_DISMAX)
"""


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
