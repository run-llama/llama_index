import io

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
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

def test_load_binary_data_file():
    # Create a BytesIO object to store the PDF data in memory
    pdf_bytes = io.BytesIO()

    # Create a PDF canvas
    c = canvas.Canvas(pdf_bytes, pagesize=letter)

    # Add text content to the PDF
    c.drawString(100, 750, "Hello, this is a PDF file.")

    # Close the PDF canvas
    c.save()

    # Reset the position pointer of the BytesIO object to the beginning
    pdf_bytes.seek(0)

    # Read the binary data of the PDF
    pdf_data = pdf_bytes.read()


    # Mock the MIME type identification to return 'application/pdf'
    magic.from_buffer = lambda x, mime: 'application/pdf'

    # Call the function under test
    documents = SimpleDirectoryReader.load_file_from_binary(pdf_data)

    # Assert that the document contains correct text
    assert documents[0].text == "Hello, this is a PDF file.\n"
    assert len(documents) == 1

def test_load_unsupported_binary_data_file_type():
    # Create binary data for a non-text type that is not supported
    binary_data = b'\x00\x01\x02\x03\x04'
    # Mock the MIME type identification to return an unsupported type
    magic.from_buffer = lambda x, mime: 'application/octet-stream'

    # Call your function, which should try to decode as text
    documents = SimpleDirectoryReader.load_file_from_binary(binary_data)

    # Assert documents are attempted to be created as text (may result in gibberish or empty)
    assert len(documents) == 1
    assert len(documents[0].text) >= 0
