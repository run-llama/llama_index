"""Data Connectors for GPT Index.

This module contains the data connectors for GPT Index. Each connector inherits
from a `BaseReader` class, connects to a data source, and loads Document objects
from that data source.

You may also choose to construct Document objects manually, for instance
in our `Insert How-To Guide <../how_to/insert.html>`_. See below for the API
definition of a Document - the bare minimum is a `text` property.

"""

from gpt_index.readers.faiss import FaissReader

# readers
from gpt_index.readers.file import SimpleDirectoryReader
from gpt_index.readers.google.gdocs import GoogleDocsReader
from gpt_index.readers.mongo import SimpleMongoReader
from gpt_index.readers.notion import NotionPageReader
from gpt_index.readers.pinecone import PineconeReader
from gpt_index.readers.schema.base import Document
from gpt_index.readers.slack import SlackReader
from gpt_index.readers.weaviate import WeaviateReader
from gpt_index.readers.wikipedia import WikipediaReader

__all__ = [
    "WikipediaReader",
    "SimpleDirectoryReader",
    "SimpleMongoReader",
    "NotionPageReader",
    "GoogleDocsReader",
    "SlackReader",
    "WeaviateReader",
    "PineconeReader",
    "FaissReader",
    "Document",
]
