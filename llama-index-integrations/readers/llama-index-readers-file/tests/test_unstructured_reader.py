import pytest

from llama_index.readers.file.unstructured import base


@pytest.mark.parametrize(
    ("url", "expected_url"),
    [(None, "http://localhost:8000"), ("https://example.invalid", "https://example.invalid")],
)
def test_from_api_configures_reader_without_loading_data(monkeypatch, url, expected_url):
    monkeypatch.setattr(base, "Element", object)

    reader = base.UnstructuredReader.from_api(api_key="test-key", url=url)

    assert reader.use_api
    assert reader.api_key == "test-key"
    assert reader.url == expected_url
