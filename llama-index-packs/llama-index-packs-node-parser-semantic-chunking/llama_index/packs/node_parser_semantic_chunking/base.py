"""
Semantic embedding chunking agent pack.

Inspired by Greg Kamradt's text splitting notebook:
https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/5_Levels_Of_Text_Splitting.ipynb

We've ported over relevant code sections. Check out the original
notebook as well!

"""

import re
from typing import Any, Dict, List, Optional

import numpy as np
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding


def combine_sentences(sentences: List[str], buffer_size: int = 1) -> List[str]:
    """
    Combine sentences.

    Ported over from:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/5_Levels_Of_Text_Splitting.ipynb

    """
    # Go through each sentence dict
    for i in range(len(sentences)):
        # Create a string that will hold the sentences which are joined
        combined_sentence = ""

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences: List[str]) -> List[float]:
    """Calculate cosine distances."""
    from sklearn.metrics.pairwise import cosine_similarity

    distances: List[float] = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["embedding"]
        embedding_next = sentences[i + 1]["embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

    # add last distance (just put 0)
    distances.append(0)

    return distances


def get_indices_above_threshold(distances: List[float], threshold: float) -> List[int]:
    """Get indices above threshold."""
    # We need to get the distance threshold that we'll consider an outlier
    # We'll use numpy .percentile() for this
    breakpoint_distance_threshold = np.percentile(
        distances, threshold
    )  # If you want more chunks, lower the percentile cutoff

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    return [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]  # The indices of those breakpoints on your list


def make_chunks(sentences: List[str], indices_above_thresh: List[int]) -> List[str]:
    """Make chunks."""
    start_index = 0
    chunks = []
    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index : end_index + 1]
        combined_text = " ".join([d["sentence"] for d in group])
        chunks.append(combined_text)

        # Update the start index for the next group
        start_index = index + 1

    # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
        chunks.append(combined_text)

    return chunks


class SemanticChunker(MetadataAwareTextSplitter):
    """
    Semantic splitter.

    Inspired by Greg's semantic chunking.

    """

    buffer_size: int = Field(
        default=1, description="Number of sentences to include in each chunk."
    )
    embed_model: Optional[BaseEmbedding] = Field(
        default=None, description="Embedding model."
    )
    breakpoint_percentile_threshold: float = Field(
        default=95.0,
        description="Percentile threshold for breakpoint distance.",
    )

    def __init__(
        self,
        buffer_size: int = 1,
        embed_model: Optional[BaseEmbedding] = None,
        breakpoint_percentile_threshold: float = 95.0,
        **kwargs: Any,
    ):
        from llama_index.embeddings.openai import OpenAIEmbedding

        super().__init__(
            buffer_size=buffer_size,
            embed_model=embed_model or OpenAIEmbedding(),
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceSplitter"

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        return self._split_text(text)

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text)

    def _split_text(self, text: str) -> List[str]:
        """
        _Split incoming text and return chunks with overlap size.

        Has a preference for complete sentences, phrases, and minimal overlap.
        """
        # Splitting the essay on '.', '?', and '!'
        single_sentences_list = re.split(r"(?<=[.?!])\s+", text)
        sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]

        combined_sentences = combine_sentences(sentences, self.buffer_size)

        # compute embeddings
        embeddings = self.embed_model.get_text_embedding_batch(
            [x["combined_sentence"] for x in combined_sentences]
        )
        # assign embeddings to the sentences
        for i, embedding in enumerate(embeddings):
            combined_sentences[i]["embedding"] = embedding

        # calculate cosine distance between adjacent sentences
        distances = calculate_cosine_distances(combined_sentences)
        for i, distance in enumerate(distances):
            combined_sentences[i]["dist_to_next"] = distance

        # get indices above threshold
        indices_above_thresh = get_indices_above_threshold(
            distances, self.breakpoint_percentile_threshold
        )

        # make chunks
        return make_chunks(combined_sentences, indices_above_thresh)


class SemanticChunkingQueryEnginePack(BaseLlamaPack):
    """
    Semantic Chunking Query Engine Pack.

    Takes in a list of documents, parses it with semantic embedding chunker,
    and runs a query engine on the resulting chunks.

    """

    def __init__(
        self,
        documents: List[Document],
        buffer_size: int = 1,
        breakpoint_percentile_threshold: float = 95.0,
    ) -> None:
        """Init params."""
        self.embed_model = OpenAIEmbedding()
        self.splitter = SemanticChunker(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=self.embed_model,
        )

        nodes = self.splitter.get_nodes_from_documents(documents)
        self.vector_index = VectorStoreIndex(nodes)
        self.query_engine = self.vector_index.as_query_engine()

    def get_modules(self) -> Dict[str, Any]:
        return {
            "vector_index": self.vector_index,
            "query_engine": self.query_engine,
            "splitter": self.splitter,
            "embed_model": self.embed_model,
        }

    def run(self, query: str) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(query)
