"""Utils file."""

import re
from typing import Optional

from transformers import GPT2TokenizerFast


def get_chunk_size_given_prompt(
    prompt: str, max_input_size: int, num_chunks: int, num_output: int
) -> int:
    """Get chunk size making sure we can also fit the prompt in."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt_tokens = tokenizer(prompt)
    num_prompt_tokens = len(prompt_tokens["input_ids"])

    return (max_input_size - num_prompt_tokens - num_output) // num_chunks


def extract_number_given_response(response: str) -> Optional[int]:
    """Extract number given the GPT-generated response."""
    numbers = re.findall(r"\d+", response)
    if len(numbers) == 0:
        return None
    else:
        return int(numbers[0])
