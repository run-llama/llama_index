"""Code splitter."""
from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.text_splitter.types import TextSplitter

DEFAULT_CHUNK_LINES = 40
DEFAULT_LINES_OVERLAP = 15
DEFAULT_MAX_CHARS = 1500


class CodeSplitter(TextSplitter):
    """Split code using a AST parser.

    Thank you to Kevin Lu / SweepAI for suggesting this elegant code splitting solution.
    https://docs.sweep.dev/blogs/chunking-2m-files
    """

    language: str = Field(
        description="The programming language of the code being split."
    )
    chunk_lines: int = Field(
        default=DEFAULT_CHUNK_LINES,
        description="The number of lines to include in each chunk.",
    )
    chunk_lines_overlap: int = Field(
        default=DEFAULT_LINES_OVERLAP,
        description="How many lines of code each chunk overlaps with.",
    )
    max_chars: int = Field(
        default=DEFAULT_MAX_CHARS, description="Maximum number of characters per chunk."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    def __init__(
        self,
        language: str,
        chunk_lines: int = 40,
        chunk_lines_overlap: int = 15,
        max_chars: int = 1500,
        callback_manager: Optional[CallbackManager] = None,
    ):
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "CodeSplitter"

    def _chunk_node(self, node: Any, text: str, last_end: int = 0) -> List[str]:
        new_chunks = []
        current_chunk = ""
        for child in node.children:
            if child.end_byte - child.start_byte > self.max_chars:
                # Child is too big, recursively chunk the child
                if len(current_chunk) > 0:
                    new_chunks.append(current_chunk)
                current_chunk = ""
                new_chunks.extend(self._chunk_node(child, text, last_end))
            elif (
                len(current_chunk) + child.end_byte - child.start_byte > self.max_chars
            ):
                # Child would make the current chunk too big, so start a new chunk
                new_chunks.append(current_chunk)
                current_chunk = text[last_end : child.end_byte]
            else:
                current_chunk += text[last_end : child.end_byte]
            last_end = child.end_byte
        if len(current_chunk) > 0:
            new_chunks.append(current_chunk)
        return new_chunks

    def split_text(self, text: str) -> List[str]:
        """Split incoming code and return chunks using the AST."""
        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            try:
                import tree_sitter_languages
            except ImportError:
                raise ImportError(
                    "Please install tree_sitter_languages to use CodeSplitter."
                )

            try:
                parser = tree_sitter_languages.get_parser(self.language)
            except Exception:
                print(
                    f"Could not get parser for language {self.language}. Check "
                    "https://github.com/grantjenks/py-tree-sitter-languages#license "
                    "for a list of valid languages."
                )
                raise

            tree = parser.parse(bytes(text, "utf-8"))

            if (
                not tree.root_node.children
                or tree.root_node.children[0].type != "ERROR"
            ):
                chunks = [
                    chunk.strip() for chunk in self._chunk_node(tree.root_node, text)
                ]
                event.on_end(
                    payload={EventPayload.CHUNKS: chunks},
                )

                return chunks
            else:
                raise ValueError(f"Could not parse code with language {self.language}.")

        # TODO: set up auto-language detection using something like https://github.com/yoeo/guesslang.
