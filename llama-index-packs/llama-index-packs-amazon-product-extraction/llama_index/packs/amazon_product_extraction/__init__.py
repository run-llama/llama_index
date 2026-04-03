import warnings

warnings.warn(
    "llama-index-packs-amazon-product-extraction is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.amazon_product_extraction.base import (
    AmazonProductExtractionPack,
)

__all__ = ["AmazonProductExtractionPack"]
