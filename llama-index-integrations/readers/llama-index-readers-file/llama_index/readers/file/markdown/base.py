"""
Markdown parser.

Contains parser for md files.

"""

import re
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MarkdownReader(BaseReader):
    """
    Markdown parser.

    Extract text from markdown files.
    Returns dictionary with keys as headers and values as the text between headers.

    """

    def __init__(
        self,
        *args: Any,
        remove_hyperlinks: bool = True,
        remove_images: bool = True,
        separator: str = " ",
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images
        self._separator = separator

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        """Convert a markdown file to a list of tuples containing header and text."""
        markdown_tups: List[Tuple[Optional[str], str]] = []
        lines = markdown_text.split("\n")

        current_lines = []
        in_code_block = False
        headers = {}
        for line in lines:
            # Toggle code block state
            if line.startswith("```"):
                in_code_block = not in_code_block

            if in_code_block:
                current_lines.append(line)
                continue
            # Process headers only when not in a code block
            else:
                line = line.strip()
                if not line:
                    continue

                header_match = re.match(r"^(#+)\s+(.*)", line)
                if header_match:
                    if current_lines and not headers:
                        # Add content before first header
                        markdown_tups.append((None, "\n".join(current_lines)))
                        current_lines.clear()
                    # Extract header level and text
                    header_level = len(
                        header_match.group(1)
                    )  # number of '#' indicates level
                    current_header = header_match.group(2)  # the header text
                    if headers.get(header_level):
                        # Add previous section to the list before switching header
                        markdown_tups.append(
                            (
                                self._separator.join(headers.values()),
                                "\n".join(current_lines),
                            )
                        )
                        # remove all headers with level greater than current header
                        headers = {k: v for k, v in headers.items() if k < header_level}
                        current_lines.clear()

                    headers[header_level] = current_header
                else:
                    current_lines.append(line)

        # Append the last section
        if current_lines or headers:
            markdown_tups.append(
                (self._separator.join(headers.values()), "\n".join(current_lines))
            )

        # Postprocess the tuples before returning
        return [
            (
                key.strip() if key else None,  # Clean up header (strip whitespace)
                re.sub(r"<.*?>", "", value),  # Remove HTML tags
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
        filepath: str,
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
        file: str,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file into string."""
        tups = self.parse_tups(file, fs=fs)
        results = []

        for header, text in tups:
            if header is None:
                results.append(Document(text=text, metadata=extra_info or {}))
            else:
                results.append(
                    Document(text=f"\n\n{header}\n{text}", metadata=extra_info or {})
                )
        return results
