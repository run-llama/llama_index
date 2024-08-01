"""Utilities for response."""

from typing import Generator


def get_response_text(response_gen: Generator) -> str:
    """Get response text."""
    response_text = ""
    for response in response_gen:
        response_text += response
    return response_text


async def aget_response_text(response_gen: Generator) -> str:
    """Get response text."""
    response_text = ""
    async for response in response_gen:
        response_text += response
    return response_text
