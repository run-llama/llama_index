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
