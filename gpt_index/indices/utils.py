"""Utilities for GPT indices."""
import re
from typing import Dict, List, Optional, Set

from gpt_index.indices.data_structs import Node
from gpt_index.utils import globals_helper


def get_sorted_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """Get sorted node list. Used by tree-strutured indices."""
    sorted_indices = sorted(node_dict.keys())
    return [node_dict[index] for index in sorted_indices]


def get_text_from_nodes(node_list: List[Node]) -> str:
    """Get text from nodes. Used by tree-structured indices."""
    return "\n".join([node.get_text() for node in node_list])


def get_numbered_text_from_nodes(node_list: List[Node]) -> str:
    """Get text from nodes in the format of a numbered list.

    Used by tree-structured indices.

    """
    text = ""
    number = 1
    for node in node_list:
        text += f"({number}) {' '.join(node.get_text().splitlines())}"
        text += "\n\n"
        number += 1
    return text


def get_chunk_size_given_prompt(
    prompt: str,
    max_input_size: int,
    num_chunks: int,
    num_output: int,
    embedding_limit: Optional[int] = None,
) -> int:
    """Get chunk size making sure we can also fit the prompt in."""
    tokenizer = globals_helper.tokenizer
    prompt_tokens = tokenizer(prompt)
    num_prompt_tokens = len(prompt_tokens["input_ids"])

    # NOTE: if embedding limit is specified, then chunk_size must not be larger than
    # embedding_limit
    result = (max_input_size - num_prompt_tokens - num_output) // num_chunks
    if embedding_limit is not None:
        return min(result, embedding_limit)
    else:
        return result


def extract_numbers_given_response(response: str, n: int = 1) -> Optional[List[int]]:
    """Extract number given the GPT-generated response.

    Used by tree-structured indices.

    """
    numbers = re.findall(r"\d+", response)
    if len(numbers) == 0:
        return None
    else:
        return numbers[:n]


def expand_tokens_with_subtokens(tokens: Set[str]) -> Set[str]:
    """Get subtokens from a list of tokens., filtering for stopwords."""
    results = set()
    for token in tokens:
        results.add(token)
        sub_tokens = re.findall(r"\w+", token)
        if len(sub_tokens) > 1:
            results.update({w for w in sub_tokens if w not in globals_helper.stopwords})

    return results


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    return text[: max_length - 3] + "..."
