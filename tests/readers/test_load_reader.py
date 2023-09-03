from typing import cast

from llama_index.readers.loading import load_reader
from llama_index.readers.notion import NotionPageReader
from llama_index.readers.string_iterable import StringIterableReader
from llama_index.readers.web import BeautifulSoupWebReader


def test_loading_readers() -> None:
    notion = NotionPageReader(integration_token="test")
    string_iterable = StringIterableReader()
    soup = BeautifulSoupWebReader(website_extractor={"test": lambda x: x})

    notion_dict = notion.to_dict()
    string_iterable_dict = string_iterable.to_dict()
    soup_dict = soup.to_dict()

    loaded_notion = cast(NotionPageReader, load_reader(notion_dict))
    loaded_string_iterable = cast(
        StringIterableReader, load_reader(string_iterable_dict)
    )
    loaded_soup = cast(BeautifulSoupWebReader, load_reader(soup_dict))

    assert loaded_notion.integration_token == notion.integration_token
    assert loaded_notion.is_remote == notion.is_remote

    assert loaded_string_iterable.is_remote == string_iterable.is_remote

    assert loaded_soup.is_remote == soup.is_remote
