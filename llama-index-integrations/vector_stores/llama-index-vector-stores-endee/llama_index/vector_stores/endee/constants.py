"""
Constants for EndeeVectorStore.

This module contains all constants used by the Endee LlamaIndex integration.
"""

from llama_index.core.vector_stores.types import FilterOperator

# Endee default constants
# These may be overridden by importing from endee.constants if available
MAX_VECTORS_PER_BATCH = 1000
DEFAULT_EF_SEARCH = 128
MAX_TOP_K_ALLOWED = 512
MAX_EF_SEARCH_ALLOWED = 1024
MAX_DIMENSION_ALLOWED = 10000
MAX_INDEX_NAME_LENGTH_ALLOWED = 48
EF_CONSTRUCTION_FIELD = "ef_con"

# Try to import constants from endee package to stay in sync
try:
    from endee.constants import (
        DEFAULT_EF_SEARCH as _DEFAULT_EF_SEARCH,
        EF_CONSTRUCTION_FIELD as _EF_CONSTRUCTION_FIELD,
        MAX_DIMENSION_ALLOWED as _MAX_DIMENSION_ALLOWED,
        MAX_EF_SEARCH_ALLOWED as _MAX_EF_SEARCH_ALLOWED,
        MAX_INDEX_NAME_LENGTH_ALLOWED as _MAX_INDEX_NAME_LENGTH_ALLOWED,
        MAX_TOP_K_ALLOWED as _MAX_TOP_K_ALLOWED,
        MAX_VECTORS_PER_BATCH as _MAX_VECTORS_PER_BATCH,
    )
    # Override defaults with values from endee package
    DEFAULT_EF_SEARCH = _DEFAULT_EF_SEARCH
    EF_CONSTRUCTION_FIELD = _EF_CONSTRUCTION_FIELD
    MAX_DIMENSION_ALLOWED = _MAX_DIMENSION_ALLOWED
    MAX_EF_SEARCH_ALLOWED = _MAX_EF_SEARCH_ALLOWED
    MAX_INDEX_NAME_LENGTH_ALLOWED = _MAX_INDEX_NAME_LENGTH_ALLOWED
    MAX_TOP_K_ALLOWED = _MAX_TOP_K_ALLOWED
    MAX_VECTORS_PER_BATCH = _MAX_VECTORS_PER_BATCH
except ImportError:
    pass

# Space types and precision types for index creation
SPACE_TYPES_VALID = ("cosine", "l2", "ip")
PRECISION_VALID = ("binary", "float16", "float32", "int16", "int8")

# Space type mapping (no aliases, direct mapping only)
SPACE_TYPE_MAP = {
    "cosine": "cosine",
    "l2": "l2",
    "ip": "ip",
}

# Vector store keys
ID_KEY = "id"
VECTOR_KEY = "values"
SPARSE_VECTOR_KEY = "sparse_values"
METADATA_KEY = "metadata"

# Batch size for add(); capped by MAX_VECTORS_PER_BATCH
DEFAULT_BATCH_SIZE = 100

# Default sparse dimension for BERT-based sparse models (SPLADE, etc.)
# This is the vocabulary size for BERT WordPiece tokenizer
DEFAULT_SPARSE_DIM = 30522

# Supported filter operations: currently only EQ and IN.
# Map FilterOperator -> endee/backend filter symbol.
SUPPORTED_FILTER_OPERATORS = (
    FilterOperator.EQ,   # eq  -> $eq
    FilterOperator.IN,   # in  -> $in
)

REVERSE_OPERATOR_MAP = {
    FilterOperator.EQ: "$eq",
    FilterOperator.IN: "$in",
}