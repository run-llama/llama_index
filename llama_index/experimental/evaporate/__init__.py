"""Evaporate.

Evaporate is an open-source project from Stanford's AI Lab:
https://github.com/HazyResearch/evaporate.
Offering techniques for structured datapoint extraction.

In the current version, we use the function generator
from a set of documents.

"""

from llama_index.experimental.evaporate.base import EvaporateExtractor

__all__ = ["EvaporateExtractor"]
