import warnings
from llama_index.readers.hive.base import HiveReader

warnings.warn(
    "Starting from v0.3.1, llama-index-hive-reader package has been deprecated due to security concerns in its SQL query handling. Use this package with caution",
    DeprecationWarning,
)


__all__ = ["HiveReader"]
