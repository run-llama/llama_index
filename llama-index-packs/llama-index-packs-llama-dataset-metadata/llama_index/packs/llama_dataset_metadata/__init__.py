import warnings

warnings.warn(
    "llama-index-packs-llama-dataset-metadata is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.llama_dataset_metadata.base import LlamaDatasetMetadataPack

__all__ = ["LlamaDatasetMetadataPack"]
