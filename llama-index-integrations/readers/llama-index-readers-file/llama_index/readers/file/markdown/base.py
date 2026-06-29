"""
Markdown parser.

Contains parser for md files.

"""

import re
from typing import Any, Dict, List, Optional, Tuple

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import yaml


_FRONTMATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n(?P<frontmatter>.*?)(?:\r?\n)---[ \t]*(?:\r?\n|$)",
    re.DOTALL,
)


class _FrontmatterLoader(yaml.SafeLoader):
    """Safe YAML loader that keeps date-like scalars as metadata strings."""


_FrontmatterLoader.yaml_implicit_resolvers = {
    key: [
        (tag, regexp)
        for tag, regexp in resolvers
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


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
        extract_frontmatter: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images
        self._separator = separator
        self._extract_frontmatter = extract_frontmatter

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

    def _read_content(
        self, filepath: str, fs: Optional[AbstractFileSystem] = None
    ) -> str:
        """Read markdown file content."""
        fs = fs or LocalFileSystem()
        with fs.open(filepath, encoding="utf-8") as f:
            return f.read().decode(encoding="utf-8")

    def _parse_content(
        self, content: str
    ) -> Tuple[List[Tuple[Optional[str], str]], Dict[str, Any]]:
        """Parse markdown content into tuples and extracted metadata."""
        content, frontmatter = self.extract_frontmatter(content)
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        if self._remove_images:
            content = self.remove_images(content)
        return self.markdown_to_tups(content), frontmatter

    def extract_frontmatter(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """Extract YAML frontmatter from markdown content."""
        if not self._extract_frontmatter:
            return content, {}

        match = _FRONTMATTER_PATTERN.match(content)
        if not match:
            return content, {}

        try:
            frontmatter = yaml.load(
                match.group("frontmatter"), Loader=_FrontmatterLoader
            ) or {}
        except yaml.YAMLError:
            return content, {}

        if not isinstance(frontmatter, dict):
            return content, {}

        return content[match.end() :], frontmatter

    def parse_tups(
        self,
        filepath: str,
        errors: str = "ignore",
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Tuple[Optional[str], str]]:
        """Parse file into tuples."""
        content = self._read_content(filepath, fs=fs)
        tups, _ = self._parse_content(content)
        return tups

    def load_data(
        self,
        file: str,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file into string."""
        content = self._read_content(file, fs=fs)
        tups, frontmatter = self._parse_content(content)
        results = []
        metadata = {**frontmatter, **(extra_info or {})}

        for header, text in tups:
            if header is None:
                results.append(Document(text=text, metadata=metadata))
            else:
                results.append(
                    Document(text=f"\n\n{header}\n{text}", metadata=metadata)
                )
        return results
