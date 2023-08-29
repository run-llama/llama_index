"""Test simple web reader."""
import string
from random import choice

import pytest

from typing import Any, Dict

from llama_index.readers import SimpleWebPageReader

try:
    import html2text
except ImportError:
    html2text = None


@pytest.mark.skipif(html2text is None, reason="html2text not installed")
def test_error_40x() -> None:
    """Test simple web reader for 40x error."""
    # Generate a random URL that doesn't exist.
    url_that_doesnt_exist = "https://{url}.{tld}"
    reader = SimpleWebPageReader()
    with pytest.raises(Exception):
        reader.load_data(
            [
                url_that_doesnt_exist.format(
                    url="".join(choice(string.ascii_lowercase) for _ in range(10)),
                    tld="".join(choice(string.ascii_lowercase) for _ in range(3)),
                )
            ]
        )


@pytest.mark.skipif(html2text is None, reason="html2text not installed")
def test_url_metadata() -> None:
    """Test simple web reader with metadata hook."""
    # Set up a reader to return the URL as metadata.
    reader = SimpleWebPageReader(metadata=lambda url: {"url": url})
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    documents = reader.load_data([url])
    assert len(documents) == 1
    assert documents[0].metadata == {"url": url}
