"""Datasets reader."""

from typing import List, Optional, Any, Iterable, Dict, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from datasets import Dataset, IterableDataset, load_dataset, Split


class DatasetsReader(BaseReader):
    """
    Datasets reader.

    Load HuggingFace datasets as documents.

    """

    @staticmethod
    def _make_document(
        sample: Dict[str, Any],
        doc_id_key: Optional[str] = None,
        text_key: Optional[str] = None,
    ) -> Document:
        kwargs = {"metadata": sample}

        if doc_id_key:
            if doc_id_key not in sample:
                msg = f"Document id key '{doc_id_key}' not found."
                raise KeyError(msg)
            kwargs["id_"] = sample[doc_id_key]

        if text_key:
            if text_key not in sample:
                msg = f"Text key '{text_key}' not found."
                raise KeyError(msg)
            kwargs["text"] = sample[text_key]

        return Document(**kwargs)

    def load_data(
        self,
        *args: Any,
        dataset: Optional[Dataset] = None,
        split: Union[Split, str] = Split.TRAIN,
        doc_id_key: Optional[str] = None,
        text_key: Optional[str] = None,
        **load_kwargs: Any,
    ) -> List[Document]:
        """
        Load data from the dataset.

        Args:
            *args: Positional arguments to pass to load_dataset.
            dataset (Optional[Dataset]): The dataset to load. load_dataset is skipped if provided. Optional.
            split (Union[Split, str]): The split to load. Default: Split.TRAIN.
            doc_id_key (Optional[str]): The key of the doc_id in samples. Optional.
            text_key (Optional[str]): The key of the text in samples. Optional.
            **load_kwargs: Keyword arguments to pass to load_dataset.

        Returns:
            List[Document]: A list of documents.

        """
        if dataset is None:
            dataset = load_dataset(*args, **load_kwargs, split=split, streaming=False)

        return [
            self._make_document(sample, doc_id_key=doc_id_key, text_key=text_key)
            for sample in dataset
        ]

    def lazy_load_data(
        self,
        *args: Any,
        dataset: Optional[IterableDataset] = None,
        split: Union[Split, str] = Split.TRAIN,
        doc_id_key: Optional[str] = None,
        text_key: Optional[str] = None,
        **load_kwargs: Any,
    ) -> Iterable[Document]:
        """
        Lazily load data from the dataset.

        Args:
            *args: Positional arguments to pass to load_dataset.
            dataset (Optional[IterableDataset]): The dataset to load. load_dataset is skipped if provided. Optional.
            split (Union[Split, str]): The split to load. Default: Split.TRAIN.
            doc_id_key (Optional[str]): The key of the doc_id in samples. Optional.
            text_key (Optional[str]): The key of the text in samples. Optional.
            **load_kwargs: Keyword arguments to pass to load_dataset.

        Returns:
            List[Document]: A list of documents.

        """
        if dataset is None:
            dataset = load_dataset(*args, **load_kwargs, split=split, streaming=True)

        # Return Document generator
        return (
            self._make_document(sample, doc_id_key=doc_id_key, text_key=text_key)
            for sample in dataset
        )
