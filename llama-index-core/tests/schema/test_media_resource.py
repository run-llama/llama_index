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


def test_hash_path_uses_posix_separator():
    """
    MediaResource.hash must use the POSIX (forward-slash) path form so
    ingestion deduplication produces the same hash on Windows and POSIX.

    We verify this by checking that the path hash is derived from the
    as_posix() string rather than str(path): on POSIX they are identical, so
    the existing test_hash constant remains valid.  On Windows str(path) would
    use backslashes and produce a different digest — this test catches that.
    """
    from hashlib import sha256 as _sha256

    path = Path("a/b/c/d.txt")
    expected_path_component = str(_sha256(path.as_posix().encode("utf-8")).hexdigest())

    resource = MediaResource(path=path)
    # The single-field hash is sha256(sha256(path_posix)) per the implementation.
    assert resource.hash == str(
        _sha256(expected_path_component.encode("utf-8")).hexdigest()
    )
    assert resource.hash != ""
