"""Common utilities and models for Azure Database for PostgreSQL operations."""

from ._base import (
    BaseAzurePGVectorStore,
)
from ._connection import (
    AzurePGConnectionPool,
    ConnectionInfo,
    check_connection,
    create_extensions,
)
from ._shared import (
    HNSW,
    Algorithm,
    BasicAuth,
    DiskANN,
    DiskANNIterativeScanMode,
    DiskANNSearchParams,
    Extension,
    HNSWIterativeScanMode,
    HNSWSearchParams,
    IVFFlat,
    IVFFlatIterativeScanMode,
    IVFFlatSearchParams,
    SSLMode,
    VectorOpClass,
    VectorType,
)
from .aio import (
    AsyncAzurePGConnectionPool,
    AsyncConnectionInfo,
    async_check_connection,
    async_create_extensions,
)

__all__ = [
    # Shared constructs
    "HNSW",
    "Algorithm",
    "BasicAuth",
    "DiskANN",
    "DiskANNIterativeScanMode",
    "DiskANNSearchParams",
    "Extension",
    "HNSWIterativeScanMode",
    "HNSWSearchParams",
    "IVFFlat",
    "IVFFlatIterativeScanMode",
    "IVFFlatSearchParams",
    "SSLMode",
    "VectorOpClass",
    "VectorType",
    # Base classes
    "BaseAzurePGVectorStore",
    # Synchronous connection constructs
    "AzurePGConnectionPool",
    "ConnectionInfo",
    "check_connection",
    "create_extensions",
    # Asynchronous connection constructs
    "AsyncAzurePGConnectionPool",
    "AsyncConnectionInfo",
    "async_check_connection",
    "async_create_extensions",
]
