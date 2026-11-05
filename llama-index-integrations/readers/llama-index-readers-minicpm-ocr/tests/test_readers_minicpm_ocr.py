"""Unit tests for MiniCPMOCRReader. The SDK client and layout model are mocked."""

from pathlib import Path
from typing import Any, Dict, List

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from PIL import Image

from llama_index.readers.minicpm_ocr import MiniCPMOCRReader
from llama_index.readers.minicpm_ocr.base import _block_image, _normalize_server_url


def test_class() -> None:
    names_of_base_classes = [base.__name__ for base in MiniCPMOCRReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_normalize_server_url() -> None:
    assert (
        _normalize_server_url(" http://localhost:8000/v1/ ") == "http://localhost:8000"
    )
    assert _normalize_server_url("http://localhost:8000/") == "http://localhost:8000"


def test_constructor_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://env.example/v1")
    monkeypatch.setenv("MINICPM_OCR_MODEL", "env-model")
    monkeypatch.setenv("MINICPM_OCR_BACKEND", "vllm")
    reader = MiniCPMOCRReader(
        base_url="http://arg.example:9/v1/",
        model="arg-model",
        backend="Ollama",
        api_key="secret",
        max_model_len=128,
        startup_timeout=2,
        request_timeout=3,
        layout_device="cuda:0",
        layout_dir="/tmp/layout",
        pdf_zoom=1.5,
    )
    assert reader.base_url == "http://arg.example:9"
    assert reader.model == "arg-model"
    assert reader.backend == "ollama"
    assert reader.api_key == "secret"
    assert reader.max_model_len == 128
    assert reader.startup_timeout == 2
    assert reader.request_timeout == 3
    assert reader.layout_device == "cuda:0"
    assert reader.layout_dir == "/tmp/layout"
    assert reader.pdf_zoom == 1.5


def test_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://localhost:8000/v1")
    for name in (
        "MINICPM_OCR_MODEL",
        "MINICPM_OCR_BACKEND",
        "MINICPM_OCR_API_KEY",
        "MINICPM_OCR_MAX_MODEL_LEN",
        "MINICPM_OCR_STARTUP_TIMEOUT",
        "MINICPM_OCR_REQUEST_TIMEOUT",
        "MINICPM_OCR_LAYOUT_DEVICE",
        "MINICPM_OCR_LAYOUT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    reader = MiniCPMOCRReader()
    assert reader.base_url == "http://localhost:8000"
    assert reader.model == "minicpm-ocr"
    assert reader.backend == "vllm"
    assert reader.api_key == ""
    assert reader.max_model_len == 40960
    assert reader.startup_timeout == 60
    assert reader.request_timeout == 300
    assert reader.layout_device == "cpu"
    assert reader.layout_dir == "~/.cache/minicpm-ocr/layout"


def test_invalid_env_int_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("MINICPM_OCR_MAX_MODEL_LEN", "lots")
    reader = MiniCPMOCRReader()
    assert reader.max_model_len == 40960


def test_missing_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINICPM_OCR_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="base_url"):
        MiniCPMOCRReader()


def test_invalid_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        MiniCPMOCRReader(base_url="http://localhost:8000", backend="nope")


def test_invalid_pdf_zoom() -> None:
    with pytest.raises(ValueError, match="pdf_zoom"):
        MiniCPMOCRReader(base_url="http://localhost:8000", pdf_zoom=0)


def test_build_config_matches_server_contract() -> None:
    reader = MiniCPMOCRReader(
        base_url="http://localhost:8000/v1",
        backend="sglang",
        api_key="tok",
        model="minicpm-ocr",
    )
    cfg = reader._build_config()
    assert cfg["engine"]["type"] == "sglang"
    assert cfg["engine"]["server_url"] == "http://localhost:8000"
    assert cfg["engine"]["model_name"] == "minicpm-ocr"
    assert cfg["engine"]["api_key"] == "tok"
    assert (
        cfg["engine"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    )
    assert cfg["layout"]["provider"] == "ppdoclayout3_paddlex"
    assert cfg["layout"]["device"] == "cpu"
    assert cfg["upright"]["enabled"] is False


class _Layout:
    def __init__(self, blocks: List[Dict[str, Any]]) -> None:
        self.blocks = blocks
        self.started = False

    def start(self) -> None:
        self.started = True

    def detect(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        return [[{"label": block["label"]} for block in self.blocks]]

    def crop(
        self, image: Image.Image, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return self.blocks


class _Engine:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def recognize(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        slice_nums: int = 0,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "slice_nums": slice_nums,
                "size": image.size,
            }
        )
        return self.responses.pop(0)


def _page() -> Image.Image:
    return Image.new("RGB", (16, 16), "white")


def _wire(
    reader: MiniCPMOCRReader,
    blocks: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
) -> _Engine:
    from minicpmvocr import DEFAULT_PROMPT_MAP

    engine = _Engine(responses)
    reader._layout = _Layout(blocks)
    reader._engine = engine
    reader._prompt_map = DEFAULT_PROMPT_MAP
    return engine


def test_recognize_page_uses_block_prompts() -> None:
    reader = MiniCPMOCRReader(base_url="http://localhost:8000")
    crop = Image.new("RGB", (8, 8), "black")
    blocks = [
        {"label": "doc_title", "image": crop, "bbox": [0, 0, 8, 8], "order": 0},
        {"label": "text", "image": crop, "bbox": [0, 8, 8, 16], "order": 1},
        {"label": "table", "image": crop, "bbox": [8, 0, 16, 8], "order": 2},
        {"label": "display_formula", "image": crop, "bbox": [8, 8, 16, 16], "order": 3},
        {"label": "header", "image": crop, "bbox": [0, 0, 4, 4], "order": 4},
        {"label": "image", "image": crop, "skip_vlm": True, "order": 5},
    ]
    engine = _wire(
        reader,
        blocks,
        [
            {"text": "Title", "finish_reason": "stop"},
            {"text": "Hello page", "finish_reason": "stop"},
            {"text": "<fcel>a<nl>", "finish_reason": "stop"},
            {"text": "E=mc^2", "finish_reason": "stop"},
        ],
    )
    text = reader._recognize_page(_page())
    prompts = [call["prompt"] for call in engine.calls]
    assert prompts == [
        "Text Recognition:",
        "Text Recognition:",
        "Table Recognition:",
        "Formula Recognition:",
    ]
    assert engine.calls[1]["max_new_tokens"] == 4096
    assert "# Title" in text
    assert "Hello page" in text
    assert "<table>" in text
    assert "E=mc^2" in text
    assert "$$" in text


def test_recognize_page_skips_truncated_empty_and_missing_crops() -> None:
    reader = MiniCPMOCRReader(base_url="http://localhost:8000")
    crop = Image.new("RGB", (4, 4), "white")
    blocks = [
        {"label": "text", "image": crop, "order": 0},
        {"label": "text", "image": crop, "order": 1},
        {"label": "text", "bbox": [5, 5, 5, 5], "order": 2},
    ]
    engine = _wire(
        reader,
        blocks,
        [
            {"text": "kept", "finish_reason": "stop"},
            {"text": "cut off", "finish_reason": "length"},
        ],
    )
    assert reader._recognize_page(_page()) == "kept"
    assert len(engine.calls) == 2


def test_block_image_falls_back_to_page_crop() -> None:
    page = Image.new("RGB", (10, 10), "white")
    crop = _block_image(page, {"bbox": [1, 2, 6, 8]})
    assert crop is not None
    assert crop.size == (5, 6)
    assert _block_image(page, {"bbox": [0, 0, 0, 4]}) is None


def test_load_data_image_and_extra_info(tmp_path: Path) -> None:
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (12, 12), "white").save(image_path)
    reader = MiniCPMOCRReader(base_url="http://localhost:8000")
    _wire(
        reader,
        [{"label": "text", "image": Image.new("RGB", (4, 4)), "order": 0}],
        [{"text": "from image", "finish_reason": "stop"}],
    )
    documents = reader.load_data(image_path, extra_info={"page": 9, "lang": "en"})
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].text == "from image"
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["file_name"] == "scan.png"
    assert documents[0].metadata["file_path"] == str(image_path)
    assert documents[0].metadata["source"] == str(image_path)
    assert documents[0].metadata["lang"] == "en"


def test_load_data_pdf_pages(tmp_path: Path) -> None:
    import pymupdf

    pdf_path = tmp_path / "pages.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.new_page()
    pdf.save(pdf_path)
    pdf.close()

    reader = MiniCPMOCRReader(base_url="http://localhost:8000", pdf_zoom=1)
    calls = {"n": 0}

    def fake_recognize(image: Image.Image) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            return ""
        return f"page {calls['n']}"

    reader._recognize_page = fake_recognize  # type: ignore[method-assign]
    documents = reader.load_data(str(pdf_path))
    assert calls["n"] == 2
    assert len(documents) == 1
    assert documents[0].text == "page 1"
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["file_name"] == "pages.pdf"


def test_load_data_rejects_missing_and_unsupported(tmp_path: Path) -> None:
    reader = MiniCPMOCRReader(base_url="http://localhost:8000")
    with pytest.raises(FileNotFoundError):
        reader.load_data(tmp_path / "missing.pdf")
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        reader.load_data(notes)


def test_ensure_runtime_starts_sdk_and_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import minicpmvocr
    import minicpmvocr.layouts.download as download

    events: List[str] = []

    class Layout:
        def start(self) -> None:
            events.append("layout")

    class Engine:
        def __init__(self, should_fail: bool) -> None:
            self.should_fail = should_fail

        def start(self) -> None:
            if self.should_fail:
                raise RuntimeError("server down")
            events.append("engine")

    state = {"fail": True}

    def fake_backend(cfg: Dict[str, Any]) -> Engine:
        assert cfg["engine"]["server_url"] == "http://localhost:8000"
        assert cfg["layout"]["model_dir"].endswith("model")
        return Engine(state["fail"])

    def fake_detector(cfg: Dict[str, Any]) -> Layout:
        assert cfg["device"] == "cpu"
        return Layout()

    monkeypatch.setattr(
        minicpmvocr, "load_prompt_map", lambda cfg: {"markdown_assembly": {}}
    )
    monkeypatch.setattr(minicpmvocr, "create_layout_detector", fake_detector)
    monkeypatch.setattr(minicpmvocr, "create_backend", fake_backend)
    monkeypatch.setattr(
        download, "ensure_layout_model", lambda directory: Path(directory) / "model"
    )

    reader = MiniCPMOCRReader(
        base_url="http://localhost:8000",
        layout_dir=str(tmp_path / "layout"),
    )
    with pytest.raises(RuntimeError, match="server down"):
        reader._ensure_runtime()
    assert reader._engine is None
    assert reader._layout is None

    state["fail"] = False
    reader._ensure_runtime()
    assert events == ["layout", "layout", "engine"]
    assert reader._engine is not None
    reader._ensure_runtime()
    assert events == ["layout", "layout", "engine"]
