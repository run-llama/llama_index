import base64
import importlib.util
import sys
import types
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

if importlib.util.find_spec("matplotlib") is None:
    _mpl = types.ModuleType("matplotlib")
    _plt = types.ModuleType("matplotlib.pyplot")
    _plt.figure = MagicMock()  # type: ignore[attr-defined]
    _plt.subplot = MagicMock()  # type: ignore[attr-defined]
    _plt.subplots = MagicMock()  # type: ignore[attr-defined]
    _plt.imshow = MagicMock()  # type: ignore[attr-defined]
    _plt.xticks = MagicMock()  # type: ignore[attr-defined]
    _plt.yticks = MagicMock()  # type: ignore[attr-defined]
    _plt.show = MagicMock()  # type: ignore[attr-defined]
    _mpl.pyplot = _plt  # type: ignore[attr-defined]
    sys.modules.setdefault("matplotlib", _mpl)
    sys.modules.setdefault("matplotlib.pyplot", _plt)

if importlib.util.find_spec("IPython") is None:
    _ipython = types.ModuleType("IPython")
    _ipython_display = types.ModuleType("IPython.display")
    _ipython_display.Markdown = MagicMock()  # type: ignore[attr-defined]
    _ipython_display.display = MagicMock()  # type: ignore[attr-defined]
    _ipython.display = _ipython_display  # type: ignore[attr-defined]
    sys.modules.setdefault("IPython", _ipython)
    sys.modules.setdefault("IPython.display", _ipython_display)

from llama_index.core.base.response.schema import Response
from llama_index.core.response.notebook_utils import (
    display_query_and_multimodal_response,
)
from llama_index.core.schema import ImageNode, NodeWithScore


def _make_b64_png() -> str:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_display_query_and_multimodal_response_metadata_without_image_nodes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-empty response.metadata without 'image_nodes' key should not raise KeyError or plt.subplots(1, 0) ValueError."""
    response = Response(response="Text-only answer", metadata={"source": "doc_1"})

    with patch("llama_index.core.response.notebook_utils.plt") as mock_plt:
        display_query_and_multimodal_response("What is in doc_1?", response)
        mock_plt.subplots.assert_not_called()
        mock_plt.show.assert_not_called()

    captured = capsys.readouterr().out
    assert "Query: What is in doc_1?" in captured
    assert "Response: Text-only answer" in captured


def test_display_query_and_multimodal_response_empty_image_nodes() -> None:
    """Empty image_nodes list (or None metadata) should skip plt.subplots(1, 0) instead of raising ValueError."""
    response = Response(response="No images retrieved", metadata={"image_nodes": []})

    with patch("llama_index.core.response.notebook_utils.plt") as mock_plt:
        display_query_and_multimodal_response("Find images", response)
        mock_plt.subplots.assert_not_called()
        mock_plt.show.assert_not_called()


def test_display_query_and_multimodal_response_supports_base64_image_node() -> None:
    """ImageNode with base64 `image` populated (and no image_path/image_url) should render via b64_2_img."""
    b64_str = _make_b64_png()
    node = ImageNode(image=b64_str)
    response = Response(
        response="Red square",
        metadata={"image_nodes": [NodeWithScore(node=node, score=0.95)]},
    )

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    with patch("llama_index.core.response.notebook_utils.plt") as mock_plt:
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        display_query_and_multimodal_response("Describe the image", response)

        mock_plt.subplots.assert_called_once_with(1, 1)
        mock_ax.imshow.assert_called_once()
        mock_plt.show.assert_called_once()


def test_display_query_and_multimodal_response_raises_when_no_image_source() -> None:
    """ImageNode with none of image_url, image_path, or image set should raise ValueError."""
    node = ImageNode()
    response = Response(
        response="Missing image",
        metadata={"image_nodes": [NodeWithScore(node=node, score=0.5)]},
    )

    with patch("llama_index.core.response.notebook_utils.plt") as mock_plt:
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        with pytest.raises(
            ValueError,
            match="A retrieved image must have image_path, image_url, or image specified.",
        ):
            display_query_and_multimodal_response("Missing image query", response)
