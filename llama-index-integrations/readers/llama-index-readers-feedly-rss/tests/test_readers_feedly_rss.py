from llama_index.core.readers.base import BaseReader
from llama_index.readers.feedly_rss import FeedlyRssReader


def test_class():
    names_of_base_classes = [b.__name__ for b in FeedlyRssReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_setup_auth_writes_utf8_token(tmp_path):
    reader = FeedlyRssReader(" token-é ")

    reader.setup_auth(directory=tmp_path)

    assert (tmp_path / "access.token").read_text(encoding="utf-8") == "token-é"
