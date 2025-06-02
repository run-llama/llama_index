from typing import Dict, List, Optional
import re

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ContentBlock, TextBlock


class SafeFormatter:
    """Safe string formatter that does not raise KeyError if key is missing."""

    def __init__(self, format_dict: Optional[Dict[str, str]] = None):
        self.format_dict = format_dict or {}

    def format(self, format_string: str) -> str:
        return re.sub(r"\{([^{}]+)\}", self._replace_match, format_string)

    def parse(self, format_string: str) -> List[str]:
        return re.findall(
            r"\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}", format_string
        )

    def _replace_match(self, match: re.Match) -> str:
        key = match.group(1)
        return str(self.format_dict.get(key, match.group(0)))


def format_string(string_to_format: str, **kwargs: str) -> str:
    """Format a string with kwargs."""
    formatter = SafeFormatter(format_dict=kwargs)
    return formatter.format(string_to_format)


def format_content_blocks(
    content_blocks: List[ContentBlock], **kwargs: str
) -> List[ContentBlock]:
    """Format content blocks with kwargs."""
    formatter = SafeFormatter(format_dict=kwargs)
    formatted_blocks: List[ContentBlock] = []
    for block in content_blocks:
        if isinstance(block, TextBlock):
            formatted_blocks.append(TextBlock(text=formatter.format(block.text)))
        else:
            formatted_blocks.append(block)

    return formatted_blocks


def get_template_vars(template_str: str) -> List[str]:
    """Get template variables from a template string."""
    variables = []
    formatter = SafeFormatter()

    for variable_name in formatter.parse(template_str):
        if variable_name:
            variables.append(variable_name)

    return variables


def is_chat_model(llm: BaseLLM) -> bool:
    return llm.metadata.is_chat_model
