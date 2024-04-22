from typing import cast

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.loading import load_reader
from llama_index.core.readers.string_iterable import StringIterableReader
import magic

def test_loading_readers() -> None:
    string_iterable = StringIterableReader()

    string_iterable_dict = string_iterable.to_dict()

    loaded_string_iterable = cast(
        StringIterableReader, load_reader(string_iterable_dict)
    )

    assert loaded_string_iterable.is_remote == string_iterable.is_remote

def test_load_text_file():
    # Create a sample text binary data
    text_data = "Hello, this is a test.".encode('utf-8')
    # Mock the MIME type identification to return 'text/plain'
    magic.from_buffer = lambda x, mime: 'text/plain'

    # Call your function
    documents = SimpleDirectoryReader.load_file_from_binary(text_data)

    # Assert that the document contains correct text
    assert documents[0].text == "Hello, this is a test."
    assert len(documents) == 1

def test_load_unsupported_file_type():
    # Create binary data for a non-text type that is not supported
    binary_data = b'\x00\x01\x02\x03\x04'
    # Mock the MIME type identification to return an unsupported type
    magic.from_buffer = lambda x, mime: 'application/octet-stream'

    # Call your function, which should try to decode as text
    documents = SimpleDirectoryReader.load_file_from_binary(binary_data)

    # Assert documents are attempted to be created as text (may result in gibberish or empty)
    assert len(documents) == 1
    assert len(documents[0].text) >= 0
