from typing import Any, AsyncGenerator, Callable, Generator, Union

from llama_index.core.llms import ChatResponseGen, ChatResponseAsyncGen, ChatResponse
from llama_index.core.memory import BaseMemory


def stream_and_write(
    memory: BaseMemory,
    stream: ChatResponseGen,
    condition_fn: Callable[[ChatResponse], bool],
    process_fn: Callable[[Any], Any] = lambda x: x,
) -> Generator[Union[bool, ChatResponse], None, None]:
    """
    Exposes a streaming generator while writing the result to memory.

    First yields the condition met status, then yields the processed chunk,
    and finally yields the full response.
    """
    full_response = None
    condition_met = False
    condition_yielded = False

    for chunk in stream:
        if not condition_met:
            condition_met = condition_fn(chunk)
            if not condition_yielded:
                yield condition_met
                condition_yielded = True
            if condition_met:
                full_response = chunk
                break

        if not condition_met:
            yield process_fn(chunk)

        full_response = chunk

    if full_response:
        memory.put(full_response)

    yield full_response


async def astream_and_write(
    memory: BaseMemory,
    stream: ChatResponseAsyncGen,
    condition_fn: Callable[[ChatResponse], bool],
    process_fn: Callable[[Any], Any] = lambda x: x,
) -> AsyncGenerator[Union[bool, ChatResponse], None]:
    """
    Exposes an async streaming generator while writing the result to memory.

    First yields the condition met status, then yields the processed chunk,
    and finally yields the full response.
    """
    full_response = None
    condition_met = False
    condition_yielded = False

    async for chunk in stream:
        if not condition_met:
            condition_met = condition_fn(chunk)
            if not condition_yielded:
                yield condition_met
                condition_yielded = True
            if condition_met:
                full_response = chunk
                break

        if not condition_met:
            yield process_fn(chunk)

        full_response = chunk

    if full_response:
        memory.put(full_response)

    yield full_response
