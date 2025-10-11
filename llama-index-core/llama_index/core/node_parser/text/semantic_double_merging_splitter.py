import re
import string
from typing import Any, Callable, Dict, List, Optional, Sequence

from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import BaseNode, Document, NodeRelationship
from llama_index.core.utils import get_tqdm_iterable, globals_helper

DEFAULT_OG_TEXT_METADATA_KEY = "original_text"

# TODO test more languages
LANGUAGES: List[str] = ["english", "german", "spanish"]
LANGUAGE_MODELS: Dict[str, List[str]] = {
    "english": ["en_core_web_md", "en_core_web_lg"],
    "german": ["de_core_news_md", "de_core_news_lg"],
    "spanish": ["es_core_news_md", "es_core_news_lg"],
}


class LanguageConfig:
    def __init__(
        self,
        language: str = "english",
        spacy_model: str = "en_core_web_md",
        model_validation: bool = True,
    ):
        if language not in LANGUAGES:
            raise ValueError(
                f"{language} language is not supported yet! Available languages: {LANGUAGES}"
            )

        if spacy_model not in LANGUAGE_MODELS[language] and model_validation:
            raise ValueError(
                f"{spacy_model} model is not matching your language: {language}"
            )

        self.language = language
        self.spacy_model = spacy_model
        self.nlp = None
        self.stopwords: List[str] = []

    def load_model(self) -> None:
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "Spacy is not installed, please install it with `pip install spacy`."
            )
        self.nlp = spacy.load(self.spacy_model)  # type: ignore
        self.stopwords = set(globals_helper.stopwords)  # type: ignore


class SemanticDoubleMergingSplitterNodeParser(NodeParser):
    """
    Semantic double merging text splitter.

    Splits a document into Nodes, with each node being a group of semantically related sentences.

    Args:
        language_config (LanguageConfig): chooses language and spacy language model to be used
        initial_threshold (float): sets threshold for initializing new chunk
        appending_threshold (float): sets threshold for appending new sentences to chunk
        merging_threshold (float): sets threshold for merging whole chunks
        max_chunk_size (int): maximum size of chunk (in characters)
        merging_range (int): How many chunks 'ahead' beyond the nearest neighbor to be merged if similar (1 or 2 available)
        merging_separator (str): The separator to use when merging chunks. Defaults to a single space.
        sentence_splitter (Optional[Callable]): splits text into sentences

    """

    language_config: LanguageConfig = Field(
        default=LanguageConfig(),
        description="Config that selects language and spacy model for chunking",
    )

    initial_threshold: float = Field(
        default=0.6,
        description=(
            "The value of semantic similarity that must be exceeded between two"
            "sentences to create a new chunk.  The bigger this "
            "value is, the more nodes will be generated. Range is from 0 to 1."
        ),
    )

    appending_threshold: float = Field(
        default=0.8,
        description=(
            "The value of semantic similarity that must be exceeded between a "
            "chunk and new sentence to add this sentence to existing chunk.  The bigger this "
            "value is, the more nodes will be generated. Range is from 0 to 1."
        ),
    )

    merging_threshold: float = Field(
        default=0.8,
        description=(
            "The value of semantic similarity that must be exceeded between two chunks "
            "to form a bigger chunk.  The bigger this value is,"
            "the more nodes will be generated. Range is from 0 to 1."
        ),
    )

    max_chunk_size: int = Field(
        default=1000,
        description="Maximum length of chunk that can be subjected to verification (number of characters)",
    )

    merging_range: int = Field(
        default=1,
        description=(
            "How many chunks 'ahead' beyond the nearest neighbor"
            "should the algorithm check during the second pass"
            "(possible options are 1 or 2"
        ),
    )

    merging_separator: str = Field(
        default=" ",
        description="The separator to use when merging chunks. Defaults to a single space.",
    )

    sentence_splitter: Callable[[str], List[str]] = Field(
        default_factory=split_by_sentence_tokenizer,
        description="The text splitter to use when splitting documents.",
        exclude=True,
    )

    @classmethod
    def class_name(cls) -> str:
        return "SemanticDoubleMergingSplitterNodeParser"

    @classmethod
    def from_defaults(
        cls,
        language_config: Optional[LanguageConfig] = LanguageConfig(),
        initial_threshold: Optional[float] = 0.6,
        appending_threshold: Optional[float] = 0.8,
        merging_threshold: Optional[float] = 0.8,
        max_chunk_size: Optional[int] = 1000,
        merging_range: Optional[int] = 1,
        merging_separator: Optional[str] = " ",
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "SemanticDoubleMergingSplitterNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()

        id_func = id_func or default_id_func

        return cls(
            language_config=language_config,
            initial_threshold=initial_threshold,
            appending_threshold=appending_threshold,
            merging_threshold=merging_threshold,
            max_chunk_size=max_chunk_size,
            merging_range=merging_range,
            merging_separator=merging_separator,
            sentence_splitter=sentence_splitter,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes."""
        # Load model
        self.language_config.load_model()

        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.build_semantic_nodes_from_nodes([node])
            all_nodes.extend(nodes)
        return all_nodes

    def build_semantic_nodes_from_documents(
        self,
        documents: Sequence[Document],
    ) -> List[BaseNode]:
        """Build window nodes from documents."""
        return self.build_semantic_nodes_from_nodes(documents)

    def build_semantic_nodes_from_nodes(
        self,
        nodes: Sequence[BaseNode],
    ) -> List[BaseNode]:
        """Build window nodes from nodes."""
        all_nodes: List[BaseNode] = []

        for node in nodes:
            text = node.get_content()
            sentences = self.sentence_splitter(text)
            sentences = [s.strip() for s in sentences]
            initial_chunks = self._create_initial_chunks(sentences)
            chunks = self._merge_initial_chunks(initial_chunks)

            split_nodes = build_nodes_from_splits(
                chunks,
                node,
                id_func=self.id_func,
            )

            previous_node: Optional[BaseNode] = None
            for split_node in split_nodes:
                if previous_node:
                    split_node.relationships[NodeRelationship.PREVIOUS] = (
                        previous_node.as_related_node_info()
                    )
                    previous_node.relationships[NodeRelationship.NEXT] = (
                        split_node.as_related_node_info()
                    )
                previous_node = split_node
            all_nodes.extend(split_nodes)

        return all_nodes

    def _create_initial_chunks(self, sentences: List[str]) -> List[str]:
        initial_chunks: List[str] = []
        chunk = sentences[0]  # ""
        new = True

        assert self.language_config.nlp is not None

        for sentence in sentences[1:]:
            if new:
                # check if 2 sentences got anything in common

                if (
                    self.language_config.nlp(
                        self._clean_text_advanced(chunk)
                    ).similarity(
                        self.language_config.nlp(self._clean_text_advanced(sentence))
                    )
                    < self.initial_threshold
                    and len(chunk) + len(sentence) + 1 <= self.max_chunk_size
                ):
                    # if not then leave first sentence as separate chunk
                    initial_chunks.append(chunk)
                    chunk = sentence
                    continue

                chunk_sentences = [chunk]
                if len(chunk) + len(sentence) + 1 <= self.max_chunk_size:
                    chunk_sentences.append(sentence)
                    chunk = self.merging_separator.join(chunk_sentences)
                    new = False
                else:
                    new = True
                    initial_chunks.append(chunk)
                    chunk = sentence
                    continue
                last_sentences = self.merging_separator.join(chunk_sentences[-2:])
                # new = False

            elif (
                self.language_config.nlp(
                    self._clean_text_advanced(last_sentences)
                ).similarity(
                    self.language_config.nlp(self._clean_text_advanced(sentence))
                )
                > self.appending_threshold
                and len(chunk) + len(sentence) + 1 <= self.max_chunk_size
            ):
                # elif nlp(last_sentences).similarity(nlp(sentence)) > self.threshold:
                chunk_sentences.append(sentence)
                last_sentences = self.merging_separator.join(chunk_sentences[-2:])
                chunk += self.merging_separator + sentence
            else:
                initial_chunks.append(chunk)
                chunk = sentence  # ""
                new = True
        initial_chunks.append(chunk)

        return initial_chunks

    def _merge_initial_chunks(self, initial_chunks: List[str]) -> List[str]:
        chunks: List[str] = []
        skip = 0
        current = initial_chunks[0]

        assert self.language_config.nlp is not None

        # TODO avoid connecting 1st chunk with 3rd if 2nd one is above some value, or if its length is above some value

        for i in range(1, len(initial_chunks)):
            # avoid connecting same chunk multiple times
            if skip > 0:
                skip -= 1
                continue

            current_nlp = self.language_config.nlp(self._clean_text_advanced(current))

            if len(current) >= self.max_chunk_size:
                chunks.append(current)
                current = initial_chunks[i]

            # check if 1st and 2nd chunk should be connected
            elif (
                current_nlp.similarity(
                    self.language_config.nlp(
                        self._clean_text_advanced(initial_chunks[i])
                    )
                )
                > self.merging_threshold
                and len(current) + len(initial_chunks[i]) + 1 <= self.max_chunk_size
            ):
                current += self.merging_separator + initial_chunks[i]

            # check if 1st and 3rd chunk are similar, if yes then merge 1st, 2nd, 3rd together
            elif (
                i <= len(initial_chunks) - 2
                and current_nlp.similarity(
                    self.language_config.nlp(
                        self._clean_text_advanced(initial_chunks[i + 1])
                    )
                )
                > self.merging_threshold
                and len(current)
                + len(initial_chunks[i])
                + len(initial_chunks[i + 1])
                + 2
                <= self.max_chunk_size
            ):
                current += (
                    self.merging_separator
                    + initial_chunks[i]
                    + self.merging_separator
                    + initial_chunks[i + 1]
                )
                skip = 1

            # check if 1st and 4th chunk are smilar, if yes then merge 1st, 2nd, 3rd and 4th together
            elif (
                i < len(initial_chunks) - 2
                and current_nlp.similarity(
                    self.language_config.nlp(
                        self._clean_text_advanced(initial_chunks[i + 2])
                    )
                )
                > self.merging_threshold
                and self.merging_range == 2
                and len(current)
                + len(initial_chunks[i])
                + len(initial_chunks[i + 1])
                + len(initial_chunks[i + 2])
                + 3
                <= self.max_chunk_size
            ):
                current += (
                    self.merging_separator
                    + initial_chunks[i]
                    + self.merging_separator
                    + initial_chunks[i + 1]
                    + self.merging_separator
                    + initial_chunks[i + 2]
                )
                skip = 2

            else:
                chunks.append(current)
                current = initial_chunks[i]

        chunks.append(current)
        return chunks

    def _clean_text_advanced(self, text: str) -> str:
        text = text.lower()
        # Remove urls
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove punctuations
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Remove stopwords
        tokens = globals_helper.punkt_tokenizer.tokenize(text)
        filtered_words = [w for w in tokens if w not in self.language_config.stopwords]

        return " ".join(filtered_words)
