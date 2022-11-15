"""Utilities for GPT indices."""
import re
from typing import Dict, List, Optional

import nltk
from nltk.corpus import stopwords
from transformers import GPT2TokenizerFast

from gpt_index.indices.data_structs import Node

nltk.download("stopwords")


def get_sorted_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """Get sorted node list. Used by tree-strutured indices."""
    sorted_indices = sorted(node_dict.keys())
    return [node_dict[index] for index in sorted_indices]


def get_text_from_nodes(node_list: List[Node]) -> str:
    """Get text from nodes. Used by tree-structured indices."""
    text = ""
    for node in node_list:
        text += node.text
        text += "\n"
    return text


def get_numbered_text_from_nodes(node_list: List[Node]) -> str:
    """Get text from nodes in the format of a numbered list.

    Used by tree-structured indices.

    """
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
    """Extract number given the GPT-generated response.

    Used by tree-structured indices.

    """
    numbers = re.findall(r"\d+", response)
    if len(numbers) == 0:
        return None
    else:
        return numbers[:n]


def extract_keywords_given_response(
    response: str, n: int = 5, lowercase: bool = True
) -> List[str]:
    """Extract keywords given the GPT-generated response.

    Used by keyword table indices.

    """
    results = []
    keywords = response.split(",")
    for k in keywords:
        if "KEYWORD" in k:
            continue
        rk = k
        if lowercase:
            rk = rk.lower()
        results.append(rk.strip())

        # if keyword consists of multiple words, split into subwords
        # (removing stopwords)
        rk_tokens = re.findall(r"\w+", rk)
        if len(rk_tokens) > 1:
            rk_tokens = [w for w in rk_tokens if w not in stopwords.words("english")]
            results.extend([rkt.strip() for rkt in rk_tokens])

    return results


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    return text[: max_length - 3] + "..."
