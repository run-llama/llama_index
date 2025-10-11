import pytest

from llama_index.readers.web import SimpleWebPageReader


@pytest.fixture()
def url() -> str:
    return "https://docs.llamaindex.ai/en/stable/module_guides/workflow/"


def test_simple_web_reader(url: str) -> None:
    documents = SimpleWebPageReader().load_data(urls=[url])
    assert len(documents) > 0
    assert isinstance(documents[0].id_, str)
    assert documents[0].id_ != url
    assert len(documents[0].id_) == 36
