from typing import Any, Callable, List, Optional, Sequence, Tuple, TypedDict
from typing_extensions import Annotated

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, SerializeAsAny, WithJsonSchema
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.node_parser.text.language_utils import (
    calculate_language_adaptive_threshold,
    detect_language,
    get_adaptive_buffer_size,
    get_multilingual_tokenizer,
)
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_OG_TEXT_METADATA_KEY = "original_text"


class SentenceCombination(TypedDict):
    sentence: str
    index: int
    combined_sentence: str
    combined_sentence_embedding: List[float]


SentenceSplitterCallable = Annotated[
    Callable[[str], List[str]],
    WithJsonSchema({"type": "string"}, mode="serialization"),
    WithJsonSchema({"type": "string"}, mode="validation"),
]


class SemanticSplitterNodeParser(NodeParser):
    """
    Semantic node parser with language-aware chunking.

    Splits a document into Nodes, with each node being a group of semantically related sentences.
    This parser automatically detects language and adapts chunking behavior accordingly.

    The parser supports automatic language detection and adaptive chunking for better results
    with multilingual text. When language_aware is enabled (default), it automatically detects
    the dominant language of each document and adjusts buffer sizes, thresholds, and tokenizers
    accordingly.

    Supported languages:
        - Chinese (zh)
        - Japanese (ja)
        - Korean (ko)
        - Arabic (ar)
        - Hebrew (he)
        - Thai (th)
        - Hindi (hi)
        - Other languages (default)

    Adaptive behavior:
        - Dense languages (Chinese/Japanese/Korean) use smaller buffer sizes (50%) and higher thresholds (+5%)
        - Right-to-left languages (Arabic/Hebrew) use smaller buffer sizes (33%) and slightly higher thresholds (+3%)
        - Other languages use standard chunking behavior

    Args:
        buffer_size (int): number of sentences to group together when evaluating semantic similarity
        embed_model: (BaseEmbedding): embedding model to use
        sentence_splitter (Optional[Callable]): splits text into sentences
        breakpoint_percentile_threshold (int): dissimilarity threshold for creating semantic breakpoints, lower value will generate more nodes
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
        language_aware (bool): whether to enable language detection and adaptive chunking (default: True)
        language_threshold_multiplier (float): additional threshold adjustment multiplier for all languages (default: 0.0)

    Example:
        >>> from llama_index.core.node_parser import SemanticSplitterNodeParser
        >>> from llama_index.embeddings.openai import OpenAIEmbedding
        >>> parser = SemanticSplitterNodeParser.from_defaults(
        ...     embed_model=OpenAIEmbedding(),
        ...     language_aware=True,  # Enable language awareness (default)
        ... )
        >>> nodes = parser.get_nodes_from_documents(documents)

    See Also:
        LANGUAGE_AWARE_CHUNKING.md for detailed documentation on language-aware chunking.

    """

    sentence_splitter: SentenceSplitterCallable = Field(
        default_factory=split_by_sentence_tokenizer,
        description="The text splitter to use when splitting documents.",
        exclude=True,
    )

    embed_model: SerializeAsAny[BaseEmbedding] = Field(
        description="The embedding model to use to for semantic comparison",
    )

    buffer_size: int = Field(
        default=1,
        description=(
            "The number of sentences to group together when evaluating semantic similarity. "
            "Set to 1 to consider each sentence individually. "
            "Set to >1 to group sentences together."
        ),
    )

    breakpoint_percentile_threshold: int = Field(
        default=95,
        description=(
            "The percentile of cosine dissimilarity that must be exceeded between a "
            "group of sentences and the next to form a node.  The smaller this "
            "number is, the more nodes will be generated"
        ),
    )

    language_aware: bool = Field(
        default=True,
        description=(
            "Whether to enable automatic language detection and adaptive chunking. "
            "When True, the parser will detect the dominant language and adjust "
            "buffer sizes and thresholds accordingly for better chunking quality."
        ),
    )

    language_threshold_multiplier: float = Field(
        default=0.0,
        description=(
            "Additional threshold adjustment for specific languages. "
            "For dense languages (Chinese, Japanese), this can be increased. "
            "Default is 0, which uses the language-aware defaults."
        ),
    )

    @classmethod
    def class_name(cls) -> str:
        return "SemanticSplitterNodeParser"

    @classmethod
    def from_defaults(
        cls,
        embed_model: Optional[BaseEmbedding] = None,
        breakpoint_percentile_threshold: Optional[int] = 95,
        buffer_size: Optional[int] = 1,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
        language_aware: bool = True,
        language_threshold_multiplier: float = 0.0,
    ) -> "SemanticSplitterNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()
        if embed_model is None:
            try:
                from llama_index.embeddings.openai import (
                    OpenAIEmbedding,
                )  # pants: no-infer-dep

                embed_model = embed_model or OpenAIEmbedding()
            except ImportError:
                raise ImportError(
                    "`llama-index-embeddings-openai` package not found, "
                    "please run `pip install llama-index-embeddings-openai`"
                )

        id_func = id_func or default_id_func

        return cls(
            embed_model=embed_model,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=buffer_size,
            sentence_splitter=sentence_splitter,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
            language_aware=language_aware,
            language_threshold_multiplier=language_threshold_multiplier,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes."""
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.build_semantic_nodes_from_documents([node], show_progress)
            all_nodes.extend(nodes)

        return all_nodes

    async def _aparse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Asynchronously parse document into nodes."""
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = await self.abuild_semantic_nodes_from_documents(
                [node], show_progress
            )
            all_nodes.extend(nodes)

        return all_nodes

    def build_semantic_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Build window nodes from documents with language awareness."""
        all_nodes: List[BaseNode] = []
        for doc in documents:
            text = doc.text

            # Use language-aware parameters if enabled
            if self.language_aware:
                language_threshold, detected_language = (
                    self._get_language_aware_threshold(text)
                )
                language_buffer_size = self._get_language_aware_buffer_size(text)
                language_tokenizer = self._get_language_aware_tokenizer(text)

                # Use language-specific settings for this document
                original_buffer_size = self.buffer_size
                original_threshold = self.breakpoint_percentile_threshold
                original_tokenizer = self.sentence_splitter

                self.buffer_size = language_buffer_size
                self.breakpoint_percentile_threshold = int(language_threshold)
                self.sentence_splitter = language_tokenizer
            else:
                detected_language = "other"

            text_splits = self.sentence_splitter(text)

            sentences = self._build_sentence_groups(text_splits)

            combined_sentence_embeddings = self.embed_model.get_text_embedding_batch(
                [s["combined_sentence"] for s in sentences],
                show_progress=show_progress,
            )

            for i, embedding in enumerate(combined_sentence_embeddings):
                sentences[i]["combined_sentence_embedding"] = embedding

            distances = self._calculate_distances_between_sentence_groups(sentences)

            chunks = self._build_node_chunks(sentences, distances)

            # Add language metadata if language awareness is enabled
            if self.language_aware:
                nodes = build_nodes_from_splits(
                    chunks,
                    doc,
                    id_func=self.id_func,
                )

                # Add language metadata to nodes
                for node in nodes:
                    if not node.metadata.get("language"):
                        node.metadata["language"] = detected_language

                # Restore original settings
                self.buffer_size = original_buffer_size
                self.breakpoint_percentile_threshold = original_threshold
                self.sentence_splitter = original_tokenizer
            else:
                nodes = build_nodes_from_splits(
                    chunks,
                    doc,
                    id_func=self.id_func,
                )

            all_nodes.extend(nodes)

        return all_nodes

    async def abuild_semantic_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Asynchronously build window nodes from documents with language awareness."""
        all_nodes: List[BaseNode] = []
        for doc in documents:
            text = doc.text

            # Use language-aware parameters if enabled
            if self.language_aware:
                language_threshold, detected_language = (
                    self._get_language_aware_threshold(text)
                )
                language_buffer_size = self._get_language_aware_buffer_size(text)
                language_tokenizer = self._get_language_aware_tokenizer(text)

                # Use language-specific settings for this document
                original_buffer_size = self.buffer_size
                original_threshold = self.breakpoint_percentile_threshold
                original_tokenizer = self.sentence_splitter

                self.buffer_size = language_buffer_size
                self.breakpoint_percentile_threshold = int(language_threshold)
                self.sentence_splitter = language_tokenizer
            else:
                detected_language = "other"

            text_splits = self.sentence_splitter(text)

            sentences = self._build_sentence_groups(text_splits)

            combined_sentence_embeddings = (
                await self.embed_model.aget_text_embedding_batch(
                    [s["combined_sentence"] for s in sentences],
                    show_progress=show_progress,
                )
            )

            for i, embedding in enumerate(combined_sentence_embeddings):
                sentences[i]["combined_sentence_embedding"] = embedding

            distances = self._calculate_distances_between_sentence_groups(sentences)

            chunks = self._build_node_chunks(sentences, distances)

            # Add language metadata if language awareness is enabled
            if self.language_aware:
                nodes = build_nodes_from_splits(
                    chunks,
                    doc,
                    id_func=self.id_func,
                )

                # Add language metadata to nodes
                for node in nodes:
                    if not node.metadata.get("language"):
                        node.metadata["language"] = detected_language

                # Restore original settings
                self.buffer_size = original_buffer_size
                self.breakpoint_percentile_threshold = original_threshold
                self.sentence_splitter = original_tokenizer
            else:
                nodes = build_nodes_from_splits(
                    chunks,
                    doc,
                    id_func=self.id_func,
                )

            all_nodes.extend(nodes)

        return all_nodes

    def _build_sentence_groups(
        self, text_splits: List[str]
    ) -> List[SentenceCombination]:
        sentences: List[SentenceCombination] = [
            {
                "sentence": x,
                "index": i,
                "combined_sentence": "",
                "combined_sentence_embedding": [],
            }
            for i, x in enumerate(text_splits)
        ]

        # Group sentences and calculate embeddings for sentence groups
        for i in range(len(sentences)):
            combined_sentence = ""

            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]["sentence"]

            combined_sentence += sentences[i]["sentence"]

            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined_sentence += sentences[j]["sentence"]

            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _calculate_distances_between_sentence_groups(
        self, sentences: List[SentenceCombination]
    ) -> List[float]:
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            similarity = self.embed_model.similarity(embedding_current, embedding_next)

            distance = 1 - similarity

            distances.append(distance)

        return distances

    def _build_node_chunks(
        self, sentences: List[SentenceCombination], distances: List[float]
    ) -> List[str]:
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_percentile_threshold
            )

            indices_above_threshold = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]

            # Chunk sentences into semantic groups based on percentile breakpoints
            start_index = 0

            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]
                combined_text = "".join([d["sentence"] for d in group])
                chunks.append(combined_text)

                start_index = index + 1

            if start_index < len(sentences):
                combined_text = "".join(
                    [d["sentence"] for d in sentences[start_index:]]
                )
                chunks.append(combined_text)
        else:
            # If, for some reason we didn't get any distances (i.e. very, very small documents) just
            # treat the whole document as a single node
            chunks = [" ".join([s["sentence"] for s in sentences])]

        return chunks

    def _get_language_aware_threshold(self, text: str) -> Tuple[float, str]:
        """
        Get an adaptive dissimilarity threshold based on detected language.

        Args:
            text: Input text

        Returns:
            Tuple of (adaptive_threshold, detected_language)

        """
        if not self.language_aware:
            return float(self.breakpoint_percentile_threshold), "other"

        detected_language = detect_language(text)
        threshold_multiplier = self.language_threshold_multiplier

        # Calculate adaptive threshold
        adaptive_threshold = calculate_language_adaptive_threshold(
            detected_language,
            base_threshold=self.breakpoint_percentile_threshold,
        )

        # Apply user-provided multiplier if specified
        if threshold_multiplier != 0.0:
            adaptive_threshold = min(99.0, adaptive_threshold + threshold_multiplier)

        return adaptive_threshold, detected_language

    def _get_language_aware_buffer_size(self, text: str) -> int:
        """
        Get an adaptive buffer size based on detected language.

        Denser languages may benefit from different buffer sizes.

        Args:
            text: Input text

        Returns:
            Adaptive buffer size

        """
        if not self.language_aware:
            return self.buffer_size

        return get_adaptive_buffer_size(
            detect_language(text),
            base_buffer_size=self.buffer_size,
            min_buffer_size=1,
            max_buffer_size=5,
        )

    def _get_language_aware_tokenizer(self, text: str) -> Callable[[str], List[str]]:
        """
        Get an appropriate tokenizer based on detected language.

        Args:
            text: Input text

        Returns:
            Appropriate tokenizer function

        """
        if not self.language_aware:
            return self.sentence_splitter

        return get_multilingual_tokenizer(
            detect_language(text),
            base_tokenizer=self.sentence_splitter,
        )
