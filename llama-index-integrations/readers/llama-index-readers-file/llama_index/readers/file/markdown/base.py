"""Markdown parser.

Contains parser for md files.

"""

import re
from pathlib import Path
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MarkdownReader(BaseReader):
    """Markdown parser.

    Extract text from markdown files.
    Returns dictionary with keys as headers and values as the text between headers.

    """

    def __init__(
        self,
        *args: Any,
        remove_hyperlinks: bool = True,
        remove_images: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        """Convert a markdown file to a dictionary.

        The keys are the headers and the values are the text under each header.

        """
        markdown_tups: List[Tuple[Optional[str], str]] = []
        lines = markdown_text.split("\n")

        current_header = None
        current_lines = []
        in_code_block = False

        for line in lines:
            if line.startswith("```"):
                # This is the end of a code block if we are already in it, and vice versa.
                in_code_block = not in_code_block

            header_match = re.match(r"^#+\s", line)
            if not in_code_block and header_match:
                # Upon first header, skip if current text chunk is empty
                if current_header is not None or len(current_lines) > 0:
                    markdown_tups.append((current_header, "\n".join(current_lines)))

                current_header = line
                current_lines.clear()
            else:
                current_lines.append(line)

        # Append final text chunk
        markdown_tups.append((current_header, "\n".join(current_lines)))

        # Postprocess the tuples before returning
        return [
            (
                key if key is None else re.sub(r"#", "", key).strip(),
                re.sub(r"<.*?>", "", value),
            )
            for key, value in markdown_tups
        ]

    def remove_images(self, content: str) -> str:
        """Remove images in markdown content but keep the description."""
        pattern = r"![(.?)](.?)"
        return re.sub(pattern, r"\1", content)

    def remove_hyperlinks(self, content: str) -> str:
        """Remove hyperlinks in markdown content."""
        pattern = r"\[(.*?)\]\((.*?)\)"
        return re.sub(pattern, r"\1", content)

    def _init_parser(self) -> Dict:
        """Initialize the parser with the config."""
        return {}

    def parse_tups(
        self,
        filepath: Path,
        errors: str = "ignore",
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Tuple[Optional[str], str]]:
        """Parse file into tuples."""
        fs = fs or LocalFileSystem()
        with fs.open(filepath, encoding="utf-8") as f:
            content = f.read().decode(encoding="utf-8")
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        if self._remove_images:
            content = self.remove_images(content)
        return self.markdown_to_tups(content)

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file into string."""
        tups = self.parse_tups(file, fs=fs)
        results = []
        # TODO: don't include headers right now
        for header, value in tups:
            if header is None:
                results.append(Document(text=value, metadata=extra_info or {}))
            else:
                results.append(
                    Document(text=f"\n\n{header}\n{value}", metadata=extra_info or {})
                )
        return results
