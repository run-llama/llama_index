import warnings
from typing import Dict, List, Optional, Callable, Sequence

from llama_index.core.bridge.pydantic import Field, model_validator
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.async_utils import run_jobs
from llama_index.core.node_parser.text.semantic_splitter import SentenceSplitterCallable
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.core.schema import BaseNode, Document
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage


# prompts taken from the paper --> SLIDE: Sliding Localized Information for Document Extraction
CONTEXT_GENERATION_SYSTEM_PROMPT = """You are an assistant which generates short English context to situate the input chunks in the input document.
Failure to adhere to this guideline will get you terminated."""

CONTEXT_GENERATION_USER_PROMPT = """Here is the document: '{window_chunk}'
Here is the chunk we want to situate within the whole document: '{chunk}'
Please give English context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with a short context.
Do not provide any additional text."""


class SlideNodeParser(NodeParser):
    """Node parser using the SLIDE based approach using LLMs to improve chunk context."""

    chunk_size: int = Field(
        default=1200,
        description="tokens per base chunk",
    )

    window_size: int = Field(
        default=11,
        description="Window size for the sliding window approach. This is the total number chunks to include in the context window, ideall an odd number.",
    )

    llm_workers: int = Field(
        default=1,
        description="Number of workers to use for LLM calls. This is only used when using the async version of the parser.",
    )

    llm: LLM = Field(description="The LLM model to use for generating local context")

    token_counter: TokenCounter = Field(description="Token counter for sentences")

    sentence_splitter: SentenceSplitterCallable = Field(
        default_factory=split_by_sentence_tokenizer,
        description="Sentence splitter to use for splitting text into sentences.",
        exclude=True,
    )

    @classmethod
    def class_name(cls) -> str:
        return "SlideNodeParser"

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = 1200,
        window_size: int = 11,
        llm_workers: int = 1,
        llm: Optional[LLM] = None,
        token_counter: Optional[TokenCounter] = None,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "SlideNodeParser":
        """Create instance of the class with default values."""
        from llama_index.core import Settings

        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func
        llm = llm or Settings.llm
        token_counter = token_counter or TokenCounter()
        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()

        return cls(
            callback_manager=callback_manager,
            id_func=id_func,
            chunk_size=chunk_size,
            window_size=window_size,
            llm_workers=llm_workers,
            llm=llm,
            token_counter=token_counter,
            sentence_splitter=sentence_splitter,
        )

    @model_validator(mode="after")
    def validate_slide_config(self):
        # 1) chunk_size ≥ 1
        if self.chunk_size < 1:
            raise ValueError("`chunk_size` must be greater than or equal to 1.")

        # 2) Warn if chunk_size is impractically small
        if self.chunk_size < 50:
            warnings.warn(
                f"chunk_size={self.chunk_size} may be too small for meaningful chunking. "
                "This could lead to poor context quality and high LLM call overhead.",
                stacklevel=2,
            )

        # 3) window_size ≥ 1
        if self.window_size < 1:
            raise ValueError("`window_size` must be greater than or equal to 1.")

        # 4) Validate LLM context budget: chunk_size × window_size
        context_window = getattr(
            getattr(self.llm, "metadata", None), "context_window", None
        )
        if context_window is not None:
            estimated_tokens = self.chunk_size * self.window_size
            if estimated_tokens > context_window:
                raise ValueError(
                    f"SLIDE configuration exceeds LLM context window: "
                    f"{self.chunk_size} × {self.window_size} = {estimated_tokens} tokens, "
                    f"but the LLM supports only {context_window} tokens."
                )
        else:
            # 5) Warn if context_window not provided
            warnings.warn(
                "The LLM does not expose `metadata.context_window`. "
                "SLIDE cannot validate token usage, which may lead to truncation or generation failures.",
                stacklevel=2,
            )

        return self

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse document into nodes."""
        # Warn if someone set llm_workers > 1 but is using sync parsing
        if self.llm_workers != 1:
            warnings.warn(
                "llm_workers has no effect when using synchronous parsing. "
                "If you want parallel LLM calls, use `aget_nodes_from_documents(...)` "
                "with llm_workers > 1.",
                stacklevel=2,
            )
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.build_slide_nodes_from_documents([node])
            all_nodes.extend(nodes)

        return all_nodes

    async def _aparse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Asynchronous parse document into nodes."""
        # If llm_workers is left at 1, no parallelism will occur.
        if self.llm_workers == 1:
            warnings.warn(
                "To parallelize LLM calls in async parsing, initialize with llm_workers > 1.",
                stacklevel=2,
            )

        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing nodes (async)"
        )

        for node in nodes_with_progress:
            nodes = await self.abuild_slide_nodes_from_documents([node], show_progress)
            all_nodes.extend(nodes)

        return all_nodes

    def create_individual_chunks(self, sentences: List[str]) -> List[str]:
        """Greedily add sentences to each chunk until we reach the chunk size limit."""
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            potential_chunk = (current_chunk + " " + sentence).strip()
            if (
                not current_chunk
                or self.token_counter.get_string_tokens(potential_chunk)
                <= self.chunk_size
            ):
                current_chunk = potential_chunk
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def build_localised_splits(
        self,
        chunks: List[str],
    ) -> List[Dict[str, str]]:
        """Generate localized context for each chunk using a sliding window approach."""
        half_window = self.window_size // 2
        localized_splits = []
        for i in range(len(chunks)):
            start = max(0, i - half_window)
            end = min(len(chunks), i + half_window + 1)
            window_chunk = " ".join(chunks[start:end])

            # format prompt with current chunk and window chunk
            llm_messages = [
                ChatMessage(role="system", content=CONTEXT_GENERATION_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=CONTEXT_GENERATION_USER_PROMPT.format(
                        window_chunk=window_chunk, chunk=chunks[i]
                    ),
                ),
            ]

            # generate localized context using LLM
            localized_context = str(self.llm.chat(messages=llm_messages))
            localized_splits.append(
                {
                    "text": chunks[i],
                    "context": localized_context,
                }
            )

        return localized_splits

    async def abuild_localised_splits(
        self,
        chunks: List[str],
        show_progress: bool = False,
    ) -> List[Dict[str, str]]:
        """Async version: batch all LLM calls for each chunk via run_jobs."""
        half_window = self.window_size // 2

        # prepare one achat() coroutine per chunk
        jobs = []
        for i, chunk in enumerate(chunks):
            start = max(0, i - half_window)
            end = min(len(chunks), i + half_window + 1)
            window_chunk = " ".join(chunks[start:end])

            llm_messages = [
                ChatMessage(role="system", content=CONTEXT_GENERATION_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=CONTEXT_GENERATION_USER_PROMPT.format(
                        window_chunk=window_chunk, chunk=chunk
                    ),
                ),
            ]
            jobs.append(self.llm.achat(messages=llm_messages))

        # run them up to a maximum of llm_workers at once, get ordered responses
        responses = await run_jobs(
            jobs=jobs,
            workers=self.llm_workers,
            show_progress=show_progress,
            desc="Generating local contexts",
        )

        # reassemble into the split format
        return [
            {"text": chunks[i], "context": str(resp)}
            for i, resp in enumerate(responses)
        ]

    def post_process_nodes(
        self,
        nodes: List[BaseNode],
        contexts: List[str],
    ) -> List[BaseNode]:
        """
        Attach slide_context metadata to each node based on the provided contexts.
        """
        for node, context in zip(nodes, contexts):
            # Preserve any existing metadata, then add our slide context
            node.metadata["local_context"] = context
        return nodes

    def build_slide_nodes_from_documents(
        self,
        documents: Sequence[Document],
    ) -> List[BaseNode]:
        """
        Build nodes enriched with localized context using a sliding window approach.
        This is the primary function of the class.
        """
        all_nodes: List[BaseNode] = []
        for document in documents:
            # Split into sentences and base chunks
            doctext = document.get_content()
            sentences = self.sentence_splitter(doctext)
            chunks = self.create_individual_chunks(sentences)

            # build localized splits
            splits = self.build_localised_splits(chunks)
            texts = [split["text"] for split in splits]
            contexts = [split["context"] for split in splits]

            # build and annotate nodes
            nodes = build_nodes_from_splits(
                text_splits=texts, document=document, id_func=self.id_func
            )

            nodes = self.post_process_nodes(nodes, contexts)
            all_nodes.extend(nodes)

        return all_nodes

    async def abuild_slide_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """
        Asynchronously build nodes enriched with localized context using a sliding window approach.
        """
        all_nodes: List[BaseNode] = []
        for document in documents:
            # Split into sentences and base chunks
            doctext = document.get_content()
            sentences = self.sentence_splitter(doctext)
            chunks = self.create_individual_chunks(sentences)

            # get localized splits using an async function
            splits = await self.abuild_localised_splits(chunks, show_progress)
            texts = [s["text"] for s in splits]
            contexts = [s["context"] for s in splits]

            # build and annotate nodes
            nodes = build_nodes_from_splits(
                text_splits=texts, document=document, id_func=self.id_func
            )

            nodes = self.post_process_nodes(nodes, contexts)
            all_nodes.extend(nodes)

        return all_nodes
