from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.text_to_image import TextToImageToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in TextToImageToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_image_downloads_set_timeout(monkeypatch):
    import sys
    import types
    from unittest.mock import MagicMock

    import requests

    calls = []

    class _Response:
        content = b"image-bytes"

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "get", fake_get)

    # show_images imports these lazily; stub them so the test needs no GUI stack.
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = MagicMock()
    pyplot.imshow = MagicMock()
    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace(open=MagicMock())
    monkeypatch.setitem(sys.modules, "PIL", pil)

    TextToImageToolSpec().show_images(["https://example.com/a.png"])

    assert calls[0].get("timeout")
