"""Experiment with different indices, models, and more."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
from langchain.input import get_color_mapping, print_text

from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.indices.vector_store import GPTSimpleVectorIndex
from gpt_index.readers.schema.base import Document

DEFAULT_INDEX_CLASSES = [GPTSimpleVectorIndex, GPTTreeIndex, GPTListIndex]
DEFAULT_MODES = ["default", "summarize", "embedding", "retrieve", "recursive"]


class Playground:
    """Experiment with indices, models, embeddings, modes, and more."""

    def __init__(self, indices: List[BaseGPTIndex], modes: List[str] = DEFAULT_MODES):
        """Initialize with indices to experiment with.

        Args:
            indices: A list of BaseGPTIndex's to experiment with
            modes: A list of modes that specify which nodes are chosen
                from the index when a query is made. A full list of modes
                available to each index can be found here:
                https://gpt-index.readthedocs.io/en/latest/reference/query.html
        """
        self._validate_indices(indices)
        self._indices = indices
        self._validate_modes(modes)
        self._modes = modes

        index_range = [str(i) for i in range(len(indices))]
        self.index_colors = get_color_mapping(index_range)

    @classmethod
    def from_docs(
        cls,
        documents: List[Document],
        index_classes: List[Type[BaseGPTIndex]] = DEFAULT_INDEX_CLASSES,
        **kwargs: Any,
    ) -> Playground:
        """Initialize with Documents using the default list of indices.

        Args:
            documents: A List of Documents to experiment with.
        """
        if len(documents) == 0:
            raise ValueError(
                "Playground must be initialized with a nonempty list of Documents."
            )

        indices = [index_class(documents) for index_class in index_classes]
        return cls(indices, **kwargs)

    def _validate_indices(self, indices: List[BaseGPTIndex]) -> None:
        """Validate a list of indices."""
        if len(indices) == 0:
            raise ValueError("Playground must have a non-empty list of indices.")
        for index in indices:
            if not isinstance(index, BaseGPTIndex):
                raise ValueError(
                    "Every index in Playground should be an instance of BaseGPTIndex."
                )

    @property
    def indices(self) -> List[BaseGPTIndex]:
        """Get Playground's indices."""
        return self._indices

    @indices.setter
    def indices(self, indices: List[BaseGPTIndex]) -> None:
        """Set Playground's indices."""
        self._validate_indices(indices)
        self._indices = indices

    def _validate_modes(self, modes: List[str]) -> None:
        """Validate a list of modes."""
        if len(modes) == 0:
            raise ValueError(
                "Playground must have a nonzero number of modes."
                "Initialize without the `modes` argument to use the default list."
            )

    @property
    def modes(self) -> List[str]:
        """Get Playground's indices."""
        return self._modes

    @modes.setter
    def modes(self, modes: List[str]) -> None:
        """Set Playground's indices."""
        self._validate_modes(modes)
        self._modes = modes

    def compare(
        self, query_text: str, to_pandas: Optional[bool] = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Compare index outputs on an input query.

        Args:
            query_text (str): Query to run all indices on.
            to_pandas (Optional[bool]): Return results in a pandas dataframe.
                True by default.

        Returns:
            The output of each index along with other data, such as the time it took to
            compute. Results are stored in a Pandas Dataframe or a list of Dicts.
        """
        print(f"\033[1mQuery:\033[0m\n{query_text}\n")
        print(f"Trying {len(self._indices) * len(self._modes)} combinations...\n\n")
        result = []
        for i, index in enumerate(self._indices):
            for mode in self._modes:
                if mode not in index.get_query_map():
                    continue
                start_time = time.time()

                index_name = type(index).__name__
                print_text(f"\033[1m{index_name}\033[0m, mode = {mode}", end="\n")
                output = index.query(query_text, mode=mode)
                print_text(str(output), color=self.index_colors[str(i)], end="\n\n")

                duration = time.time() - start_time

                result.append(
                    {
                        "Index": index_name,
                        "Mode": mode,
                        "Output": str(output),
                        "Duration": duration,
                        "LLM Tokens": index.llm_predictor.last_token_usage,
                        "Embedding Tokens": index.embed_model.last_token_usage,
                    }
                )
        print(f"\nRan {len(result)} combinations in total.")

        if to_pandas:
            return pd.DataFrame(result)
        else:
            return result
