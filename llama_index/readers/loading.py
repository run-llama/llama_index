from typing import Any, Dict, Type

from llama_index.readers.base import BasePydanticReader
from llama_index.readers.discord_reader import DiscordReader
from llama_index.readers.elasticsearch import ElasticsearchReader
from llama_index.readers.google_readers.gdocs import GoogleDocsReader
from llama_index.readers.google_readers.gsheets import GoogleSheetsReader
from llama_index.readers.notion import NotionPageReader
from llama_index.readers.slack import SlackReader
from llama_index.readers.string_iterable import StringIterableReader
from llama_index.readers.twitter import TwitterTweetReader
from llama_index.readers.web import (
    BeautifulSoupWebReader,
    RssReader,
    SimpleWebPageReader,
    TrafilaturaWebReader,
)
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader

ALL_READERS: Dict[str, Type[BasePydanticReader]] = {
    DiscordReader.class_name(): DiscordReader,
    ElasticsearchReader.class_name(): ElasticsearchReader,
    GoogleDocsReader.class_name(): GoogleDocsReader,
    GoogleSheetsReader.class_name(): GoogleSheetsReader,
    NotionPageReader.class_name(): NotionPageReader,
    SlackReader.class_name(): SlackReader,
    StringIterableReader.class_name(): StringIterableReader,
    TwitterTweetReader.class_name(): TwitterTweetReader,
    SimpleWebPageReader.class_name(): SimpleWebPageReader,
    TrafilaturaWebReader.class_name(): TrafilaturaWebReader,
    RssReader.class_name(): RssReader,
    BeautifulSoupWebReader.class_name(): BeautifulSoupWebReader,
    WikipediaReader.class_name(): WikipediaReader,
    YoutubeTranscriptReader.class_name(): YoutubeTranscriptReader,
}


def load_reader(data: Dict[str, Any]) -> BasePydanticReader:
    if isinstance(data, BasePydanticReader):
        return data
    class_name = data.get("class_name", None)
    if class_name is None:
        raise ValueError("Must specify `class_name` in reader data.")

    if class_name not in ALL_READERS:
        raise ValueError(f"Reader class name {class_name} not found.")

    # remove static attribute
    data.pop("is_remote", None)

    return ALL_READERS[class_name].from_dict(data)
