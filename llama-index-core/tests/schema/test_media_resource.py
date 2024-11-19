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
    png_1px = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    m = MediaResource(data=png_1px.encode("utf-8"), mimetype=None)
    assert m.mimetype == "image/png"


def test_hash():
    data = b"test bytes"
    path = Path(__file__).resolve().parent / "data" / "data.txt"
    url = AnyUrl("http://example.com")
    assert (
        MediaResource(data=data, path=path, url=url, text="some text").hash
        == "30ffd5d92992d12d59991a97a6da08fec784c8cb527f34a96c1cdc43edcdb661"
    )
    assert MediaResource().hash == ""
