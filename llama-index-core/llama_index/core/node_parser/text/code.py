"""Code splitter."""

from typing import Any, Callable, List, Literal, Optional, Tuple

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.schema import Document
from llama_index.core.utils import get_tokenizer

DEFAULT_CHUNK_LINES = 40
DEFAULT_LINES_OVERLAP = 15
DEFAULT_MAX_CHARS = 1500
DEFAULT_MAX_TOKENS = 512


class CodeSplitter(TextSplitter):
    """
    Split code using a AST parser.

    Thank you to Kevin Lu / SweepAI for suggesting this elegant code splitting solution.
    https://docs.sweep.dev/blogs/chunking-2m-files

    Supports both character-based and token-based chunking modes for more precise
    control over chunk sizes when working with language models.
    """

    language: str = Field(
        description="The programming language of the code being split."
    )
    chunk_lines: int = Field(
        default=DEFAULT_CHUNK_LINES,
        description="The number of lines to include in each chunk.",
        gt=0,
    )
    chunk_lines_overlap: int = Field(
        default=DEFAULT_LINES_OVERLAP,
        description="How many lines of code each chunk overlaps with.",
        ge=0,
    )
    max_chars: int = Field(
        default=DEFAULT_MAX_CHARS,
        description="Maximum number of characters per chunk.",
        gt=0,
    )
    count_mode: Literal["token", "char"] = Field(
        default="char",
        description="Mode for counting chunk size: 'char' for characters, 'token' for tokens.",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Maximum number of tokens per chunk (used when count_mode='token').",
        gt=0,
    )
    _parser: Any = PrivateAttr()
    _tokenizer: Callable = PrivateAttr()

    def __init__(
        self,
        language: str,
        chunk_lines: int = DEFAULT_CHUNK_LINES,
        chunk_lines_overlap: int = DEFAULT_LINES_OVERLAP,
        max_chars: int = DEFAULT_MAX_CHARS,
        count_mode: Literal["token", "char"] = "char",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tokenizer: Optional[Callable] = None,
        parser: Any = None,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> None:
        """
        Initialize a CodeSplitter.

        Args:
            language: The programming language of the code being split.
            chunk_lines: The number of lines to include in each chunk.
            chunk_lines_overlap: How many lines of code each chunk overlaps with.
            max_chars: Maximum number of characters per chunk.
            count_mode: Mode for counting chunk size: 'char' for characters, 'token' for tokens.
            max_tokens: Maximum number of tokens per chunk (used when count_mode='token').
            tokenizer: Optional tokenizer function for token-based counting.
            parser: Optional tree-sitter Parser object.
            callback_manager: Optional callback manager.
            include_metadata: Whether to include metadata in chunks.
            include_prev_next_rel: Whether to include previous/next relationships.
            id_func: Optional function to generate chunk IDs.

        """
        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func

        super().__init__(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
            count_mode=count_mode,
            max_tokens=max_tokens,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )

        # Initialize tokenizer if using token mode
        self._tokenizer = tokenizer or get_tokenizer()

        if parser is None:
            try:
                import tree_sitter_language_pack  # pants: no-infer-dep

                parser = tree_sitter_language_pack.get_parser(language)  # type: ignore
            except ImportError:
                raise ImportError(
                    "Please install tree_sitter_language_pack to use CodeSplitter."
                    "Or pass in a parser object."
                )
            except Exception:
                print(
                    f"Could not get parser for language {language}. Check "
                    "https://github.com/Goldziher/tree-sitter-language-pack?tab=readme-ov-file#available-languages "
                    "for a list of valid languages."
                )
                raise

        if not hasattr(parser, "parse"):
            raise ImportError(
                "The installed version of tree-sitter-language-pack is not compatible. "
                "Please install a compatible version: "
                "pip install 'tree-sitter-language-pack<1.0'"
            )

        self._parser = parser

    @classmethod
    def from_defaults(
        cls,
        language: str,
        chunk_lines: int = DEFAULT_CHUNK_LINES,
        chunk_lines_overlap: int = DEFAULT_LINES_OVERLAP,
        max_chars: int = DEFAULT_MAX_CHARS,
        count_mode: Literal["token", "char"] = "char",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tokenizer: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        parser: Any = None,
    ) -> "CodeSplitter":
        """Create a CodeSplitter with default values."""
        return cls(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
            count_mode=count_mode,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            callback_manager=callback_manager,
            parser=parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "CodeSplitter"

    def _chunk_node(
        self, node: Any, text_bytes: bytes, last_end: int = 0
    ) -> List[Tuple[str, bool]]:
        """
        Recursively chunk a node into smaller pieces based on size and line limits.

        Args:
            node (Any): The AST node to chunk.
            text_bytes (bytes): The original source code text as bytes.
            last_end (int, optional): The ending position of the last processed chunk. Defaults to 0.

        Returns:
            List[Tuple[str, bool]]: Code chunks and whether the chunk starts
                after a line-limited boundary.

        """
        new_chunks: List[Tuple[str, bool]] = []
        current_chunk = ""
        current_chunk_line_limited = False
        max_size = self.max_chars if self.count_mode == "char" else self.max_tokens
        # Reserve room for overlap in every non-overlapping chunk.  Applying the
        # overlap after the AST traversal keeps recursive chunking simple while
        # ensuring the final chunks still honor ``chunk_lines``.
        max_chunk_lines = max(1, self.chunk_lines - self.chunk_lines_overlap)

        for child in node.children:
            child_text = text_bytes[child.start_byte : child.end_byte].decode("utf-8")
            child_size = (
                len(child_text)
                if self.count_mode == "char"
                else len(self._tokenizer(child_text))
            )
            child_line_count = len(child_text.splitlines())

            if child_size > max_size or child_line_count > max_chunk_lines:
                # Child is too big, recursively chunk the child
                if len(current_chunk) > 0:
                    new_chunks.append((current_chunk, current_chunk_line_limited))
                current_chunk = ""
                current_chunk_line_limited = False
                if child.children:
                    child_chunks = self._chunk_node(child, text_bytes, last_end)
                else:
                    # Leaf nodes have no sub-structure to recurse into, so split
                    # their text directly to preserve both size and line limits.
                    child_text = text_bytes[last_end : child.end_byte].decode("utf-8")
                    child_chunks = self._split_oversized_leaf(
                        child_text, max_size, max_chunk_lines
                    )

                if child_chunks:
                    # The first chunk starts after a line-limited boundary when
                    # the child itself exceeded the line budget.  Preserve the
                    # flag so overlap can be applied at this boundary without
                    # applying it to size-only splits inside the child.
                    child_chunks[0] = (
                        child_chunks[0][0],
                        child_line_count > max_chunk_lines,
                    )
                    new_chunks.extend(child_chunks)
            else:
                # Calculate what adding this child would do to current chunk size
                child_segment = text_bytes[last_end : child.end_byte].decode("utf-8")
                new_chunk_text = current_chunk + child_segment
                new_chunk_size = (
                    len(new_chunk_text)
                    if self.count_mode == "char"
                    else len(self._tokenizer(new_chunk_text))
                )
                new_chunk_line_count = len(new_chunk_text.strip().splitlines())

                if new_chunk_size > max_size or new_chunk_line_count > max_chunk_lines:
                    # Child would make the current chunk too big, so start a new chunk
                    if len(current_chunk) > 0:
                        new_chunks.append((current_chunk, current_chunk_line_limited))
                    current_chunk = child_segment
                    current_chunk_line_limited = new_chunk_line_count > max_chunk_lines
                else:
                    current_chunk += child_segment
            last_end = child.end_byte

        if len(current_chunk) > 0:
            new_chunks.append((current_chunk, current_chunk_line_limited))
        return new_chunks

    def _split_oversized_leaf(
        self, text: str, max_size: int, max_lines: int
    ) -> List[Tuple[str, bool]]:
        """
        Split text from a leaf node that exceeds a size or line limit.

        A leaf AST node (such as a long string literal or comment) has no
        children to recurse into, so its text is split directly to keep the
        content within the size limit instead of dropping it.

        Args:
            text (str): The leaf node text to split.
            max_size (int): The maximum chunk size, in characters or tokens
                depending on ``count_mode``.
            max_lines (int): The maximum number of lines per non-overlapping
                chunk.

        Returns:
            List[Tuple[str, bool]]: Chunks and whether each chunk starts after
                a complete line boundary. A piece that cannot be reduced further
                (e.g. a single character whose token count already exceeds
                ``max_size``) is kept as-is rather than dropped.

        """
        if not text:
            return []

        # Split at line boundaries first so a multi-line leaf (for example a
        # block comment or string) cannot bypass the line limit.  The resulting
        # pieces are then split by the active size limit as needed.
        line_chunks: List[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            candidate = current + line
            if current and len(candidate.splitlines()) > max_lines:
                line_chunks.append(current)
                current = line
            else:
                current = candidate
        if current:
            line_chunks.append(current)

        chunks: List[Tuple[str, bool]] = []
        previous_line_split_by_size = False
        for line_index, line_chunk in enumerate(line_chunks):
            if self.count_mode == "char":
                pieces = [
                    line_chunk[i : i + max_size]
                    for i in range(0, len(line_chunk), max_size)
                ]
            else:
                # Token mode: greedily accumulate characters while staying
                # within the token budget. This keeps the split reversible
                # (decoding token ids is not guaranteed for an arbitrary
                # tokenizer).
                pieces = []
                current = ""
                for char in line_chunk:
                    if current and len(self._tokenizer(current + char)) > max_size:
                        pieces.append(current)
                        current = char
                    else:
                        current += char
                if current:
                    pieces.append(current)

            for piece_index, piece in enumerate(pieces):
                starts_after_line_boundary = (
                    line_index > 0
                    and piece_index == 0
                    and not previous_line_split_by_size
                )
                chunks.append((piece, starts_after_line_boundary))
            previous_line_split_by_size = len(pieces) > 1
        return chunks

    def _chunk_size(self, text: str) -> int:
        """Return the active size metric for a chunk."""
        if self.count_mode == "char":
            return len(text)
        return len(self._tokenizer(text))

    def _apply_line_overlap(self, chunks: List[Tuple[str, bool]]) -> List[str]:
        """Add configured line overlap without violating chunk limits."""
        if len(chunks) < 2 or self.chunk_lines_overlap == 0:
            return [chunk.strip() for chunk, _ in chunks]

        overlapped_chunks = [chunks[0][0].strip()]
        for raw_chunk, line_limited in chunks[1:]:
            chunk = raw_chunk.strip()
            if not line_limited:
                overlapped_chunks.append(chunk)
                continue

            previous_lines = overlapped_chunks[-1].splitlines(keepends=True)
            current_lines = chunk.splitlines(keepends=True)
            max_overlap = min(
                self.chunk_lines_overlap,
                len(previous_lines),
                max(0, self.chunk_lines - len(current_lines)),
            )

            while max_overlap > 0:
                prefix = "".join(previous_lines[-max_overlap:])
                if prefix and not prefix.endswith(("\n", "\r")):
                    prefix += "\n"
                candidate = prefix + chunk
                if self._chunk_size(candidate) <= (
                    self.max_chars if self.count_mode == "char" else self.max_tokens
                ):
                    chunk = candidate
                    break
                max_overlap -= 1

            overlapped_chunks.append(chunk)

        return overlapped_chunks

    def split_text(self, text: str) -> List[str]:
        """
        Split incoming code into chunks using the AST parser.

        This method parses the input code into an AST and then chunks it while preserving
        syntactic structure. Supports both character-based and token-based chunking modes
        for more precise control over chunk sizes.

        Args:
            text (str): The source code text to split.

        Returns:
            List[str]: A list of code chunks that respect size limits based on count_mode.

        Raises:
            ValueError: If the code cannot be parsed for the specified language.

        """
        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            text_bytes = bytes(text, "utf-8")
            tree = self._parser.parse(text_bytes)

            if (
                not tree.root_node.children
                or tree.root_node.children[0].type != "ERROR"
            ):
                chunks = self._apply_line_overlap(
                    self._chunk_node(tree.root_node, text_bytes)
                )
                event.on_end(
                    payload={EventPayload.CHUNKS: chunks},
                )

                return chunks
            else:
                raise ValueError(f"Could not parse code with language {self.language}.")

        # TODO: set up auto-language detection using something like https://github.com/yoeo/guesslang.
