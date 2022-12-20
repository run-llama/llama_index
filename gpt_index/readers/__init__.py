"""Data Connectors for GPT Index.

This module contains the data connectors for GPT Index. Each connector inherits
from a `BaseReader` class, connects to a data source, and loads BaseDocument objects
from that data source.

"""

from gpt_index.readers.faiss import FaissReader

# readers
from gpt_index.readers.file import SimpleDirectoryReader
from gpt_index.readers.google.gdocs import GoogleDocsReader
from gpt_index.readers.mongo import SimpleMongoReader
from gpt_index.readers.notion import NotionPageReader
from gpt_index.readers.pinecone import PineconeReader
from gpt_index.readers.slack import SlackReader
from gpt_index.readers.weaviate.reader import WeaviateReader
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
]
