"""Markdown parser.

Contains parser for md files.

"""
import re
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

from gpt_index.readers.file.base_parser import BaseParser


class MarkdownParser(BaseParser):
    """Markdown parser.
    Extract text from markdown files.
    Returns dictionary with keys as headers and values as the text between headers.
    """

    def __init__(
        self,
        parser_config: Optional[Dict] = {
            "remove_hyperlinks": True,
            "remove_images": True,
        },
    ):
        """Init params."""
        self._parser_config = parser_config

    def init_parser(self) -> None:
        """Init parser and store it."""
        parser_config = self._init_parser()
        self._parser_config = parser_config

    def markdown_to_dict(self, markdown_text: str) -> Dict[Optional[str], str]:
        """Convert a markdown file to a dictionary. The keys are the headers and the values are the text under each header."""
        markdown_dict = {}
        lines = markdown_text.split("\n")

        current_header = None
        current_text = ""

        for line in lines:
            header_match = re.match(r"^#+\s", line)
            if header_match:
                if current_header is not None:
                    if current_text == "" or None:
                        continue
                    markdown_dict[current_header] = current_text

                current_header = line
                current_text = ""
            else:
                current_text += line + "\n"
        markdown_dict[current_header] = current_text

        if current_header is not None:
            markdown_dict = {
                re.sub(r"#", "", key).strip(): re.sub(r"<.*?>", "", value)
                for key, value in markdown_dict.items()
            }
        else:
            markdown_dict = {
                key: re.sub("\n", "", value) for key, value in markdown_dict.items()
            }

        return markdown_dict

    def remove_images(self, content: str) -> str:
        """Get a dictionary of a markdown file from its path"""
        pattern = r"!{1}\[\[(.*)\]\]"
        content = re.sub(pattern, "", content)
        return content

    def remove_hyperlinks(self, content: str) -> str:
        """Get a dictionary of a markdown file from its path"""
        pattern = r"\[(.*?)\]\((.*?)\)"
        content = re.sub(pattern, r"\1", content)
        return content

    @property
    def parser_config_set(self) -> bool:
        """Check if parser config is set."""
        return self._parser_config is not None

    @property
    def parser_config(self) -> Dict:
        """Check if parser config is set."""
        if self._parser_config is None:
            raise ValueError("Parser config not set.")
        return self._parser_config

    def _init_parser(self) -> Dict:
        """Initialize the parser with the config."""
        return {}

    def parse_file(
        self, filepath: Path, errors: str = "ignore"
    ) -> Dict[Optional[str], str]:
        """Parse file."""
        with open(filepath, "r") as f:
            content = f.read()
        if self._parser_config["remove_hyperlinks"]:
            content = self.remove_hyperlinks(content)
        if self._parser_config["remove_images"]:
            content = self.remove_images(content)
        markdown_dict = self.markdown_to_dict(content)
        return markdown_dict
