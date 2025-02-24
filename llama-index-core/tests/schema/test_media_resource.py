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
    assert MediaResource().hash == ""
