"""Header-aware Markdown node parser with token-limit enforcement."""

import re
from typing import Any, Callable, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tokenizer, get_tqdm_iterable

_DEFAULT_CHUNK_SIZE = 1024
_HEADER_RE = re.compile(r"^(#+)\s(.*)")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


class HeaderAwareMarkdownSplitter(NodeParser):
    """Markdown node parser that respects header groupings *and* enforces token limits.

    Unlike :class:`MarkdownNodeParser` (which has no size constraint) or
    :class:`SentenceSplitter` (which has no markdown awareness), this parser
    keeps each header together with its body text whenever the section fits
    within ``chunk_size``.  Sections that exceed the limit are split at
    paragraph / sentence boundaries with the header context **prepended** to
    every sub-chunk so retrieval never loses the structural context.

    Args:
        chunk_size: Maximum number of tokens per chunk.
        header_path_separator: Separator for the ``header_path`` metadata value.
        include_header_in_chunks: Prepend the header hierarchy to each sub-chunk
            when a section must be split.
        sub_splitter: Optional callable ``(text, chunk_size) -> list[str]`` used
            to split oversized section bodies.  Defaults to a simple
            paragraph-then-sentence splitter.  Swap this out for e.g. a semantic
            splitter to get embedding-aware splits.
    """

    chunk_size: int = Field(
        default=_DEFAULT_CHUNK_SIZE,
        description="Maximum number of tokens per chunk.",
        gt=0,
    )
    header_path_separator: str = Field(
        default="/",
        description="Separator for the header_path metadata value.",
    )
    include_header_in_chunks: bool = Field(
        default=True,
        description="Prepend the header hierarchy to each sub-chunk when a section is split.",
    )

    _tokenizer: Callable = PrivateAttr()
    _sub_splitter: Optional[Callable[..., List[str]]] = PrivateAttr()

    def __init__(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        header_path_separator: str = "/",
        include_header_in_chunks: bool = True,
        tokenizer: Optional[Callable] = None,
        sub_splitter: Optional[Callable[..., List[str]]] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable] = None,
    ) -> None:
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            chunk_size=chunk_size,
            header_path_separator=header_path_separator,
            include_header_in_chunks=include_header_in_chunks,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )
        self._tokenizer = tokenizer or get_tokenizer()
        self._sub_splitter = sub_splitter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Split a single document node into header-aware chunks."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        sections = self._parse_markdown_sections(text)
        result: List[TextNode] = []

        for header_line, body, header_path in sections:
            section_text = (header_line + "\n" + body).strip() if header_line else body.strip()
            if not section_text:
                continue

            token_count = len(self._tokenizer(section_text))

            if token_count <= self.chunk_size:
                result.append(self._build_node(section_text, node, header_path))
            else:
                # Section exceeds chunk_size — sub-split the body.
                sub_chunks = self._split_oversized_section(
                    header_line, body.strip(), header_path
                )
                for chunk in sub_chunks:
                    result.append(self._build_node(chunk, node, header_path))

        return result

    # ------------------------------------------------------------------
    # Markdown parsing
    # ------------------------------------------------------------------

    def _parse_markdown_sections(
        self, text: str
    ) -> List[tuple[str, str, str]]:
        """Parse markdown into ``(header_line, body_text, header_path)`` tuples.

        Returns one tuple per section.  ``header_line`` is the raw ``## Foo``
        line (empty string for content before the first header).
        ``header_path`` is the slash-separated ancestor path *excluding* the
        current header (matching :class:`MarkdownNodeParser` convention).
        """
        lines = text.split("\n")
        sections: List[tuple[str, str, str]] = []
        header_stack: List[tuple[int, str]] = []
        current_header_line = ""
        current_body_lines: List[str] = []
        code_block = False

        def _flush() -> None:
            body = "\n".join(current_body_lines)
            path = self.header_path_separator.join(h[1] for h in header_stack[:-1]) if header_stack else ""
            sections.append((current_header_line, body, path))

        for line in lines:
            # Track code blocks to avoid treating ```# Foo``` as a header.
            if _FENCE_RE.match(line.lstrip()):
                code_block = not code_block
                current_body_lines.append(line)
                continue

            if not code_block:
                header_match = _HEADER_RE.match(line)
                if header_match:
                    # Flush previous section.  Must happen BEFORE modifying
                    # header_stack because _flush reads header_stack[:-1]
                    # to build the ancestor path for the previous section.
                    if current_header_line or current_body_lines:
                        _flush()

                    level = len(header_match.group(1))
                    header_text = header_match.group(2)

                    while header_stack and header_stack[-1][0] >= level:
                        header_stack.pop()
                    header_stack.append((level, header_text))

                    current_header_line = line
                    current_body_lines = []
                    continue

            current_body_lines.append(line)

        # Final section.
        if current_header_line or current_body_lines:
            _flush()

        return sections

    # ------------------------------------------------------------------
    # Sub-splitting
    # ------------------------------------------------------------------

    def _split_oversized_section(
        self, header_line: str, body: str, header_path: str
    ) -> List[str]:
        """Split an oversized section body into chunks, prepending header context."""
        prefix = ""
        if self.include_header_in_chunks and header_line:
            prefix = header_line + "\n"

        prefix_tokens = len(self._tokenizer(prefix)) if prefix else 0
        available = self.chunk_size - prefix_tokens
        if available < 1:
            available = 1

        if self._sub_splitter is not None:
            raw_chunks = self._sub_splitter(body, available)
        else:
            raw_chunks = self._default_split(body, available)

        return [(prefix + chunk).strip() for chunk in raw_chunks if chunk.strip()]

    def _default_split(self, text: str, max_tokens: int) -> List[str]:
        """Split text by paragraphs first, then sentences, respecting max_tokens."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para

            if len(self._tokenizer(candidate)) <= max_tokens:
                current_chunk = candidate
            else:
                # Flush what we have.
                if current_chunk:
                    chunks.append(current_chunk)

                # If this single paragraph fits, start a new chunk.
                if len(self._tokenizer(para)) <= max_tokens:
                    current_chunk = para
                else:
                    # Paragraph itself is too big — split by sentences.
                    sentence_chunks = self._split_by_sentences(para, max_tokens)
                    if not sentence_chunks:
                        # Fallback: emit the paragraph even if oversized.
                        chunks.append(para)
                        current_chunk = ""
                    else:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_sentences(self, text: str, max_tokens: int) -> List[str]:
        """Last-resort split by sentence boundaries.

        Uses a simple regex that handles ``. ``, ``! ``, ``? `` and newlines.
        Sentences that still exceed *max_tokens* are further split on word
        boundaries so the chunk-size guarantee is never silently violated.
        """
        sentences = re.split(r"(?<=[.!?])\s+|\n", text)
        chunks: List[str] = []
        current = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # If a single sentence exceeds the limit, split on whitespace.
            if len(self._tokenizer(sent)) > max_tokens:
                if current:
                    chunks.append(current)
                    current = ""
                for word in sent.split():
                    candidate = (current + " " + word).strip() if current else word
                    if len(self._tokenizer(candidate)) <= max_tokens:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = word
                continue

            candidate = (current + " " + sent).strip() if current else sent
            if len(self._tokenizer(candidate)) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sent

        if current:
            chunks.append(current)

        return chunks

    # ------------------------------------------------------------------
    # Node construction
    # ------------------------------------------------------------------

    def _build_node(
        self, text: str, source_node: BaseNode, header_path: str
    ) -> TextNode:
        """Build a TextNode with header_path metadata."""
        node = build_nodes_from_splits([text], source_node, id_func=self.id_func)[0]

        if self.include_metadata:
            sep = self.header_path_separator
            node.metadata["header_path"] = (
                sep + header_path + sep if header_path else sep
            )

        return node

    # ------------------------------------------------------------------
    # NodeParser interface
    # ------------------------------------------------------------------

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            all_nodes.extend(self.get_nodes_from_node(node))

        return all_nodes
