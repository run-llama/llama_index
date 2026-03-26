"""Preprocess Reader."""

from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PreprocessReader(BaseReader):
    """
    Preprocess reader.

    This reader has been discontinued. The Preprocess service is no longer
    available and the ``pypreprocess`` package is no longer maintained.
    Please remove this dependency from your projects.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The Preprocess service has been discontinued and is permanently"
            " unavailable. Please remove llama-index-readers-preprocess from"
            " your dependencies."
        )

    def load_data(self, **kwargs) -> List[Document]:
        raise RuntimeError(
            "The Preprocess service has been discontinued and is permanently"
            " unavailable."
        )
