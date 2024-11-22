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
    png_1px = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    m = MediaResource(data=png_1px.encode("utf-8"), mimetype=None)
    assert m.mimetype == "image/png"


def test_hash():
    assert (
        MediaResource(
            data=b"test bytes",
            path="foo/bar/baz",
            url=AnyUrl("http://example.com"),
            text="some text",
        ).hash
        == "7ac964db7843a9ffb37cda7b5b9822b0f84111d6a271b4991dd26d1fc68490d3"
    )
    assert MediaResource().hash == ""
