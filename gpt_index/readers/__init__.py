"""Data Connectors for LlamaIndex.

This module contains the data connectors for LlamaIndex. Each connector inherits
from a `BaseReader` class, connects to a data source, and loads Document objects
from that data source.

You may also choose to construct Document objects manually, for instance
in our `Insert How-To Guide <../how_to/insert.html>`_. See below for the API
definition of a Document - the bare minimum is a `text` property.

"""

from gpt_index.readers.chroma import ChromaReader
from gpt_index.readers.discord_reader import DiscordReader
from gpt_index.readers.elasticsearch import ElasticsearchReader
from gpt_index.readers.faiss import FaissReader

# readers
from gpt_index.readers.file.base import SimpleDirectoryReader
from gpt_index.readers.github_readers.github_repository_reader import (
    GithubRepositoryReader,
)
from gpt_index.readers.google_readers.gdocs import GoogleDocsReader
from gpt_index.readers.make_com.wrapper import MakeWrapper
from gpt_index.readers.mbox import MboxReader
from gpt_index.readers.mongo import SimpleMongoReader
from gpt_index.readers.notion import NotionPageReader
from gpt_index.readers.obsidian import ObsidianReader
from gpt_index.readers.pinecone import PineconeReader
from gpt_index.readers.qdrant import QdrantReader
from gpt_index.readers.schema.base import Document
from gpt_index.readers.slack import SlackReader
from gpt_index.readers.string_iterable import StringIterableReader
from gpt_index.readers.twitter import TwitterTweetReader
from gpt_index.readers.weaviate.reader import WeaviateReader
from gpt_index.readers.web import (
    BeautifulSoupWebReader,
    RssReader,
    SimpleWebPageReader,
    TrafilaturaWebReader,
)
from gpt_index.readers.wikipedia import WikipediaReader
from gpt_index.readers.youtube_transcript import YoutubeTranscriptReader

__all__ = [
    "WikipediaReader",
    "YoutubeTranscriptReader",
    "SimpleDirectoryReader",
    "SimpleMongoReader",
    "NotionPageReader",
    "GoogleDocsReader",
    "DiscordReader",
    "SlackReader",
    "WeaviateReader",
    "PineconeReader",
    "QdrantReader",
    "ChromaReader",
    "FaissReader",
    "Document",
    "StringIterableReader",
    "SimpleWebPageReader",
    "BeautifulSoupWebReader",
    "TrafilaturaWebReader",
    "RssReader",
    "MakeWrapper",
    "TwitterTweetReader",
    "ObsidianReader",
    "GithubRepositoryReader",
    "MboxReader",
    "ElasticsearchReader",
]
