import warnings

warnings.warn(
    "llama-index-packs-raft-dataset is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.raft_dataset.base import RAFTDatasetPack

__all__ = ["RAFTDatasetPack"]
