"""Code splitter."""

from typing import Any, Callable, List, Literal, Optional

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
        gt=0,
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
        from tree_sitter import Parser  # pants: no-infer-dep

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
        if not isinstance(parser, Parser):
            raise ValueError("Parser must be a tree-sitter Parser object.")

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

    def _chunk_node(self, node: Any, text_bytes: bytes, last_end: int = 0) -> List[str]:
        """
        Recursively chunk a node into smaller pieces based on character or token limits.

        Args:
            node (Any): The AST node to chunk.
            text_bytes (bytes): The original source code text as bytes.
            last_end (int, optional): The ending position of the last processed chunk. Defaults to 0.

        Returns:
            List[str]: A list of code chunks that respect the size limits.

        """
        new_chunks = []
        current_chunk = ""
        max_size = self.max_chars if self.count_mode == "char" else self.max_tokens

        for child in node.children:
            child_text = text_bytes[child.start_byte : child.end_byte].decode("utf-8")
            child_size = (
                len(child_text)
                if self.count_mode == "char"
                else len(self._tokenizer(child_text))
            )

            if child_size > max_size:
                # Child is too big, recursively chunk the child
                if len(current_chunk) > 0:
                    new_chunks.append(current_chunk)
                current_chunk = ""
                new_chunks.extend(self._chunk_node(child, text_bytes, last_end))
            else:
                # Calculate what adding this child would do to current chunk size
                new_chunk_text = current_chunk + text_bytes[
                    last_end : child.end_byte
                ].decode("utf-8")
                new_chunk_size = (
                    len(new_chunk_text)
                    if self.count_mode == "char"
                    else len(self._tokenizer(new_chunk_text))
                )

                if new_chunk_size > max_size:
                    # Child would make the current chunk too big, so start a new chunk
                    if len(current_chunk) > 0:
                        new_chunks.append(current_chunk)
                    current_chunk = text_bytes[last_end : child.end_byte].decode(
                        "utf-8"
                    )
                else:
                    current_chunk += text_bytes[last_end : child.end_byte].decode(
                        "utf-8"
                    )
            last_end = child.end_byte

        if len(current_chunk) > 0:
            new_chunks.append(current_chunk)
        return new_chunks

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
                chunks = [
                    chunk.strip()
                    for chunk in self._chunk_node(tree.root_node, text_bytes)
                ]
                event.on_end(
                    payload={EventPayload.CHUNKS: chunks},
                )

                return chunks
            else:
                raise ValueError(f"Could not parse code with language {self.language}.")

        # TODO: set up auto-language detection using something like https://github.com/yoeo/guesslang.
