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
    path = Path(__file__).parent / "data" / "data.txt"
    url = AnyUrl("http://example.com")
    assert (
        MediaResource(data=data, path=path, url=url).hash
        == "4be66ea6f5222861df37e88d4635bffb99e183435f79fba13055b835b5dc420b"
    )
    assert (
        MediaResource(path=path, url=url).hash
        == "c0f5efbef0fe98aa90619444250b1a5eb23158d6686f0b190838f3d544ec85b9"
    )
    assert (
        MediaResource(url=url).hash
        == "2a1b402420ef46577471cdc7409b0fa2c6a204db316e59ade2d805435489a067"
    )
    assert MediaResource().hash == ""
