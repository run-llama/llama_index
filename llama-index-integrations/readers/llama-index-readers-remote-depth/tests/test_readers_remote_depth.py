import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.remote_depth import RemoteDepthReader


def test_class():
    names_of_base_classes = [b.__name__ for b in RemoteDepthReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


@pytest.mark.parametrize(
    ("depth", "expected_urls"),
    [
        (0, ["https://example.com/root"]),
        (1, ["https://example.com/root", "https://example.com/child"]),
    ],
)
def test_load_data_respects_link_depth(monkeypatch, depth, expected_urls):
    root_url = "https://example.com/root"
    child_url = "https://example.com/child"
    grandchild_url = "https://example.com/grandchild"
    links = {
        root_url: [child_url],
        child_url: [grandchild_url],
        grandchild_url: [],
    }
    loaded_urls = []

    class RecordingRemoteReader:
        def __init__(self, **kwargs):
            pass

        def load_data(self, url):
            loaded_urls.append(url)
            return []

    reader = RemoteDepthReader(depth=depth)
    monkeypatch.setattr(reader, "get_links", links.__getitem__)
    monkeypatch.setattr(
        "llama_index.readers.remote_depth.base.RemoteReader", RecordingRemoteReader
    )

    reader.load_data(root_url)

    assert loaded_urls == expected_urls
