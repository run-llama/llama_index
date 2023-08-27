"""Code splitter."""
from typing import Any, List, Optional

try:
    from pydantic.v1 import Field
except ImportError:
    from pydantic import Field

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
        description="The programming languge of the code being split."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    def __init__(
        self,
        language: str,
        callback_manager: Optional[CallbackManager] = None,
    ):
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            language=language,
            callback_manager=callback_manager,
        )

    def _chunk_node(self, node: Any, text: str, last_end: int = 0, context_list: Optional[List[str]] = None) -> List[str]:
        if context_list is None:
            context_list = []

        new_chunks = []
        context_str = ''.join(context_list)
        current_chunk = context_str  # Initialize current_chunk with current context

        for child in node.children:

            # Add the new signature or header to the context list before recursing
            new_context_list = context_list.copy()
            if len(child.children) > 0 and child.children[-1].type == 'block':
                # Get only the 'signature' or 'header' of the new context.
                # In python, last_end will represent the spaces before child
                # In any language, since child.children[-1] is a block, child.children[-2] will be the end of the signature
                new_context = text[last_end:child.children[-2].end_byte]
                new_context_list.append(new_context)
                next_chunks = self._chunk_node(child.children[-1], text, child.children[-2].end_byte, new_context_list)
                new_chunks.extend(next_chunks)
            else:
                current_chunk += text[last_end:child.end_byte]

            last_end = child.end_byte

        if len(current_chunk) > len(context_str):  # If current_chunk has more than just the context
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
            except Exception as e:
                print(
                    f"Could not get parser for language {self.language}. Check "
                    "https://github.com/grantjenks/py-tree-sitter-languages#license "
                    "for a list of valid languages."
                )
                raise e

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
