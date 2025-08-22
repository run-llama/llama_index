from pathlib import Path

from llama_index.core.bridge.pydantic import AnyUrl
from llama_index.core.schema import MediaResource


def test_defaults():
    m = MediaResource()
    assert m.data is None
    assert m.embeddings is None
    assert m.mimetype is None
    assert m.path is None
    assert m.url is None


def test_mimetype():
    png_1px = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    m = MediaResource(data=png_1px, mimetype=None)
    assert m.mimetype == "image/png"


def test_mimetype_raw_data():
    import requests

    resp = requests.get(
        "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"
    )
    m = MediaResource(data=resp.content)
    assert m.mimetype == "image/jpeg"


def test_mimetype_from_path():
    m = MediaResource(path=Path("my-image.jpg"), mimetype=None)
    assert m.mimetype == "image/jpeg"


def test_mimetype_prioritizes_data():
    png_1px = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    m = MediaResource(
        data=png_1px.encode("utf-8"), mimetype=None, path=Path("my_image.jpg")
    )
    assert m.mimetype == "image/png"


def test_hash():
    assert (
        MediaResource(
            data=b"test bytes",
            path=Path("foo/bar/baz"),
            url=AnyUrl("http://example.com"),
            text="some text",
        ).hash
        == "04414a5f03ad7fa055229b4d3690d47427cb0b65bc7eb8f770d1ecbd54ab4909"
    )
    # Test that empty MediaResource returns None
    assert MediaResource().hash is None

    # Test that MediaResource with empty content still generates a hash
    assert MediaResource(text="").hash is not None
    assert MediaResource(text="").hash != ""

    # Test that MediaResource with None values but some content generates a hash
    assert MediaResource(text="", data=b"").hash is not None
    assert MediaResource(text="", data=b"").hash != ""


def test_hash_edge_cases():
    """Test various edge cases for hash generation."""
    # Test with only text (empty string)
    empty_text = MediaResource(text="")
    assert empty_text.hash is not None
    assert empty_text.hash != ""

    # Test with only data (empty bytes)
    empty_data = MediaResource(data=b"")
    assert empty_data.hash is not None
    assert empty_data.hash != ""

    # Test with only path
    path_only = MediaResource(path=Path("test.txt"))
    assert path_only.hash is not None
    assert path_only.hash != ""

    # Test with only URL
    url_only = MediaResource(url=AnyUrl("http://example.com"))
    assert url_only.hash is not None
    assert url_only.hash != ""

    # Test with mixed None and empty values
    mixed_empty = MediaResource(text="", data=None, path=None, url=None)
    assert mixed_empty.hash is not None
    assert mixed_empty.hash != ""

    # Test with all None values
    all_none = MediaResource(text=None, data=None, path=None, url=None)
    assert all_none.hash is None


def test_hash_consistency():
    """Test that hash values are consistent for the same content."""
    # Same content should produce same hash
    resource1 = MediaResource(text="hello", data=b"world")
    resource2 = MediaResource(text="hello", data=b"world")
    assert resource1.hash == resource2.hash

    # Different content should produce different hashes
    resource3 = MediaResource(text="hello", data=b"different")
    assert resource1.hash != resource3.hash

    # Empty resources should all return None
    empty1 = MediaResource()
    empty2 = MediaResource()
    assert empty1.hash is None
    assert empty2.hash is None
