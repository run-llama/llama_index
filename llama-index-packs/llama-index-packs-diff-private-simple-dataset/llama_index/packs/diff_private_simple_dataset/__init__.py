import warnings

warnings.warn(
    "llama-index-packs-diff-private-simple-dataset is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.diff_private_simple_dataset.base import (
    DiffPrivateSimpleDatasetPack,
)


__all__ = ["DiffPrivateSimpleDatasetPack"]
