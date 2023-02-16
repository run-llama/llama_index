"""Experiment with different indices, models, and more."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from langchain.input import get_color_mapping, print_text

from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.indices.vector_store.simple import GPTSimpleVectorIndex
from gpt_index.readers.schema.base import Document

DEFAULT_INDEX_CLASSES = [GPTSimpleVectorIndex, GPTTreeIndex, GPTListIndex]
DEFAULT_MODES = ["default", "summarize", "embedding", "retrieve", "recursive"]


class Playground:
    """Experiment with indices, model, embeddings, modes, and more."""

    def __init__(
        self, indices: List[BaseGPTIndex], modes: Optional[List[str]] = DEFAULT_MODES
    ):
        """Initialize with indices to experiment with.

        Args:
            indices: A List of BaseGPTIndex's to experiment with
        """
        self.update_indices(indices)
        self.update_modes(modes)

        index_range = [str(i) for i in range(len(indices))]
        self.index_colors = get_color_mapping(index_range)

    @classmethod
    def from_docs(cls, documents: List[Document], **kwargs: Any) -> Playground:
        """Initialize with Documents to easily test them across the default list of indices.

        Args:
            documents: A List of Documents to experiment with.
        """
        if len(documents) == 0:
            raise ValueError(
                "Playground must be initialized with a nonempty list of Documents."
            )

        indices = [index_class(documents) for index_class in DEFAULT_INDEX_CLASSES]
        return cls(indices, **kwargs)

    def update_indices(self, indices: List[BaseGPTIndex]) -> None:
        """Update Playground's indices."""
        if len(indices) == 0:
            raise ValueError("Playground must have a non-empty list of indices.")
        for index in indices:
            if not isinstance(index, BaseGPTIndex):
                raise ValueError(
                    "Every index in Playground should be an instance of BaseGPTIndex."
                )

        self.indices = indices

    def update_modes(self, modes: List[str]):
        """Update Playground's query modes."""
        if len(modes) == 0:
            raise ValueError(
                "Playground must have a nonzero number of modes. Initialize without the `modes` argument to use the default list."
            )

        self.modes = modes

    def compare(
        self, query_text: str, to_pandas: Optional[bool] = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Compare index outputs on an input query.

        Args:
            query_text (str): Query to run all indices on.
            to_pandas (Optional[bool]): Return results in a pandas dataframe. True by default.

        Returns:
            The output of each index along with other data, such as the time it took to compute. Results are stored in a Pandas Dataframe or a list of Dicts.
        """
        print(f"\033[1mQuery:\033[0m\n{query_text}\n")
        print(f"Trying {len(self.indices) * len(self.modes)} combinations...\n\n")
        result = []
        for i, index in enumerate(self.indices):
            for mode in self.modes:
                if mode not in index.get_query_map():
                    continue
                start_time = time.time()

                index_name = type(index).__name__
                print_text(f"\033[1m{index_name}\033[0m, mode = {mode}", end="\n")
                output = index.query(query_text, mode=mode)
                print_text(output, color=self.index_colors[str(i)], end="\n\n")

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
