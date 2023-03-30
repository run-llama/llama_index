import itertools
from typing import Iterable, List


def chunk_iterable(iterable: Iterable, batch_size: int) -> List[tuple]:
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunks = []
    chunk = tuple(itertools.islice(it, batch_size))
    for chunk in iter(lambda: tuple(itertools.islice(it, batch_size)), ()):
        chunks.append(chunk)
    return chunks
