"""Utilities for GPT indices."""
import re
from typing import Dict, List, Optional

from transformers import GPT2TokenizerFast

from gpt_index.indices.data_structs import Node


def get_sorted_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """Get sorted node list."""
    sorted_indices = sorted(node_dict.keys())
    return [node_dict[index] for index in sorted_indices]


def get_text_from_nodes(node_list: List[Node]) -> str:
    """Get text from nodes."""
    text = ""
    for node in node_list:
        text += node.text
        text += "\n"
    return text


def get_numbered_text_from_nodes(node_list: List[Node]) -> str:
    """Get text from nodes in the format of a numbered list."""
    text = ""
    number = 1
    for node in node_list:
        text += f"({number}) {' '.join(node.text.splitlines())}"
        text += "\n\n"
        number += 1
    return text


def get_chunk_size_given_prompt(
    prompt: str, max_input_size: int, num_chunks: int, num_output: int
) -> int:
    """Get chunk size making sure we can also fit the prompt in."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt_tokens = tokenizer(prompt)
    num_prompt_tokens = len(prompt_tokens["input_ids"])

    return (max_input_size - num_prompt_tokens - num_output) // num_chunks


def extract_numbers_given_response(response: str, n: int = 1) -> Optional[List[int]]:
    """Extract number given the GPT-generated response."""
    numbers = re.findall(r"\d+", response)
    if len(numbers) == 0:
        return None
    else:
        return numbers[:n]
