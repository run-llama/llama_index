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
    # Test that MediaResource with content generates a hash (hash will be different with new implementation)
    full_resource = MediaResource(
        data=b"test bytes",
        path=Path("foo/bar/baz"),
        url=AnyUrl("http://example.com"),
        text="some text",
    )
    assert full_resource.hash is not None
    assert full_resource.hash != ""
    assert isinstance(full_resource.hash, str)
    assert len(full_resource.hash) == 64  # SHA256 hex string length

    # Test that empty MediaResource ALWAYS returns a hash (never None)
    assert MediaResource().hash is not None
    assert MediaResource().hash != ""
    assert isinstance(MediaResource().hash, str)

    # Test that MediaResource with empty content generates a different hash than None
    empty_text_resource = MediaResource(text="")
    none_text_resource = MediaResource(text=None)
    assert empty_text_resource.hash != none_text_resource.hash

    # Test that MediaResource with mixed empty/None generates a hash
    mixed_resource = MediaResource(text="", data=b"")
    assert mixed_resource.hash is not None
    assert mixed_resource.hash != ""


def test_hash_edge_cases():
    """Test various edge cases for hash generation."""
    # Test with only text (empty string)
    empty_text = MediaResource(text="")
    assert empty_text.hash is not None
    assert empty_text.hash != ""
    assert isinstance(empty_text.hash, str)

    # Test with only data (empty bytes)
    empty_data = MediaResource(data=b"")
    assert empty_data.hash is not None
    assert empty_data.hash != ""
    assert isinstance(empty_data.hash, str)

    # Test with only path
    path_only = MediaResource(path=Path("test.txt"))
    assert path_only.hash is not None
    assert path_only.hash != ""
    assert isinstance(path_only.hash, str)

    # Test with only URL
    url_only = MediaResource(url=AnyUrl("http://example.com"))
    assert url_only.hash is not None
    assert url_only.hash != ""
    assert isinstance(url_only.hash, str)

    # Test with mixed None and empty values
    mixed_empty = MediaResource(text="", data=None, path=None, url=None)
    assert mixed_empty.hash is not None
    assert mixed_empty.hash != ""
    assert isinstance(mixed_empty.hash, str)

    # Test with all None values - NOW ALWAYS RETURNS A HASH
    all_none = MediaResource(text=None, data=None, path=None, url=None)
    assert all_none.hash is not None
    assert all_none.hash != ""
    assert isinstance(all_none.hash, str)


def test_hash_consistency():
    """Test that hash values are consistent for the same content."""
    # Same content should produce same hash
    resource1 = MediaResource(text="hello", data=b"world")
    resource2 = MediaResource(text="hello", data=b"world")
    assert resource1.hash == resource2.hash

    # Different content should produce different hashes
    resource3 = MediaResource(text="hello", data=b"different")
    assert resource1.hash != resource3.hash

    # Empty resources should all return the SAME hash (not None)
    empty1 = MediaResource()
    empty2 = MediaResource()
    assert empty1.hash == empty2.hash
    assert empty1.hash is not None
    assert empty2.hash is not None


def test_hash_none_vs_empty_distinction():
    """Test the key requirement: None vs empty string distinction."""
    # This is what the reviewer wanted - different hashes for None vs empty
    resource_none = MediaResource(text=None)
    resource_empty = MediaResource(text="")

    # These should be different hashes
    assert resource_none.hash != resource_empty.hash
    assert resource_none.hash is not None
    assert resource_empty.hash is not None

    # Test with data field too
    data_none = MediaResource(data=None)
    data_empty = MediaResource(data=b"")
    assert data_none.hash != data_empty.hash

    # Test completely empty vs partially empty
    completely_empty = MediaResource()  # All None
    text_empty_only = MediaResource(text="")  # text empty, others None
    assert completely_empty.hash != text_empty_only.hash
