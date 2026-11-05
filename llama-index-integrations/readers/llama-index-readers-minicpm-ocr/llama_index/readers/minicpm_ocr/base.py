"""MiniCPM-OCR reader."""

import logging
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
_BACKENDS = ("vllm", "sglang", "openai", "llamacpp", "ollama")
_TABLE_LABELS = ("table", "chart")

_DEFAULT_MODEL = "minicpm-ocr"
_DEFAULT_BACKEND = "vllm"
_DEFAULT_MAX_MODEL_LEN = 40960
_DEFAULT_STARTUP_TIMEOUT = 60
_DEFAULT_REQUEST_TIMEOUT = 300
_DEFAULT_LAYOUT_DEVICE = "cpu"
_DEFAULT_LAYOUT_DIR = "~/.cache/minicpm-ocr/layout"
_DEFAULT_PDF_ZOOM = 2.0


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _normalize_server_url(url: str) -> str:
    """The SDK expects the server root, without a trailing /v1."""
    normalized = (url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    return normalized


def _resize_to_min_side(
    image: Image.Image, target_min_side: Optional[int]
) -> Image.Image:
    if not target_min_side:
        return image
    width, height = image.size
    if min(width, height) >= target_min_side:
        return image
    scale = target_min_side / float(min(width, height))
    return image.resize(
        (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
    )


def _looks_like_table_markup(content: str) -> bool:
    return any(token in content for token in ("<fcel>", "<ecel>", "<nl>"))


def _block_image(page: Image.Image, block: Dict[str, Any]) -> Optional[Image.Image]:
    """Return the layout crop, or a page crop when the detector only returned a box."""
    crop = block.get("image")
    if crop is not None and getattr(crop, "size", 0):
        if isinstance(crop, Image.Image):
            return crop.convert("RGB")
        return Image.fromarray(crop).convert("RGB")

    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return page.crop((x1, y1, x2, y2)).convert("RGB")


class MiniCPMOCRReader(BaseReader):
    """
    Read PDFs and images into per-page Markdown documents with MiniCPM-OCR.

    PP-DocLayoutV3 layout detection runs locally through the ``minicpm-ocr`` SDK.
    Each crop is recognized by an external MiniCPM-OCR server (vLLM, SGLang,
    OpenAI-compatible, llama.cpp, or Ollama). Model weights are not bundled.

    Constructor arguments override the matching ``MINICPM_OCR_*`` environment
    variables. Layout detection and recognition stay single-threaded.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        max_model_len: Optional[int] = None,
        startup_timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
        layout_device: Optional[str] = None,
        layout_dir: Optional[str] = None,
        pdf_zoom: float = _DEFAULT_PDF_ZOOM,
    ) -> None:
        """Configure the external recognition server and the local layout detector."""
        resolved_url = _normalize_server_url(
            base_url if base_url is not None else _env("MINICPM_OCR_BASE_URL")
        )
        if not resolved_url:
            raise ValueError(
                "MiniCPM-OCR base_url is required. Pass base_url or set "
                "MINICPM_OCR_BASE_URL to the server root, for example "
                "http://localhost:8000 (without /v1)."
            )

        resolved_backend = (
            (
                backend
                if backend is not None
                else _env("MINICPM_OCR_BACKEND", _DEFAULT_BACKEND)
            )
            .strip()
            .lower()
        )
        if resolved_backend not in _BACKENDS:
            raise ValueError(
                f"Unsupported MiniCPM-OCR backend {resolved_backend!r}. "
                f"Choose one of: {', '.join(_BACKENDS)}."
            )

        resolved_model = (
            model if model is not None else _env("MINICPM_OCR_MODEL", _DEFAULT_MODEL)
        ).strip()
        if not resolved_model:
            raise ValueError("MiniCPM-OCR model name is required.")
        if pdf_zoom <= 0:
            raise ValueError("pdf_zoom must be positive.")

        self.base_url = resolved_url
        self.model = resolved_model
        self.backend = resolved_backend
        self.api_key = api_key if api_key is not None else _env("MINICPM_OCR_API_KEY")
        self.max_model_len = (
            max_model_len
            if max_model_len is not None
            else _env_int("MINICPM_OCR_MAX_MODEL_LEN", _DEFAULT_MAX_MODEL_LEN)
        )
        self.startup_timeout = (
            startup_timeout
            if startup_timeout is not None
            else _env_int("MINICPM_OCR_STARTUP_TIMEOUT", _DEFAULT_STARTUP_TIMEOUT)
        )
        self.request_timeout = (
            request_timeout
            if request_timeout is not None
            else _env_int("MINICPM_OCR_REQUEST_TIMEOUT", _DEFAULT_REQUEST_TIMEOUT)
        )
        self.layout_device = (
            layout_device
            if layout_device is not None
            else _env("MINICPM_OCR_LAYOUT_DEVICE", _DEFAULT_LAYOUT_DEVICE)
        ).strip() or _DEFAULT_LAYOUT_DEVICE
        self.layout_dir = (
            layout_dir
            if layout_dir is not None
            else _env("MINICPM_OCR_LAYOUT_DIR", _DEFAULT_LAYOUT_DIR)
        )
        self.pdf_zoom = pdf_zoom

        self._prompt_map: Optional[Dict[str, Any]] = None
        self._layout: Any = None
        self._engine: Any = None
        self._lock = threading.Lock()

    def _build_config(self) -> Dict[str, Any]:
        from minicpmvocr import DEFAULT_CONFIG

        cfg = deepcopy(DEFAULT_CONFIG)
        cfg["engine"].update(
            {
                "type": self.backend,
                "model_name": self.model,
                "server_url": self.base_url,
                "api_key": self.api_key,
                "max_model_len": self.max_model_len,
                "startup_timeout_sec": self.startup_timeout,
                "request_timeout_sec": self.request_timeout,
                # Output belongs in message content. With thinking left on, some
                # servers move the transcription into reasoning_content.
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        )
        cfg["layout"].update(
            {
                "provider": "ppdoclayout3_paddlex",
                "model_dir": "",
                "device": self.layout_device,
            }
        )
        # Rotation correction downloads extra models and is outside this reader.
        cfg["upright"]["enabled"] = False
        return cfg

    def _ensure_runtime(self) -> None:
        if (
            self._engine is not None
            and self._layout is not None
            and self._prompt_map is not None
        ):
            return

        from minicpmvocr import create_backend, create_layout_detector, load_prompt_map
        from minicpmvocr.layouts.download import ensure_layout_model

        cfg = self._build_config()
        prompt_map = load_prompt_map(cfg)
        layout_dir = Path(self.layout_dir).expanduser()
        cfg["layout"]["model_dir"] = str(ensure_layout_model(layout_dir))
        layout = create_layout_detector(cfg["layout"])
        layout.start()
        engine = create_backend(cfg)
        engine.start()
        # Publish only after startup succeeds so a failed start is retried.
        self._prompt_map = prompt_map
        self._layout = layout
        self._engine = engine

    def _iter_pages(self, path: Path) -> Iterable[Tuple[int, Image.Image]]:
        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            with Image.open(path) as image:
                yield 1, image.convert("RGB")
            return
        if suffix != ".pdf":
            supported = ", ".join(sorted(_IMAGE_SUFFIXES | {".pdf"}))
            raise ValueError(
                f"Unsupported file type {suffix or '(none)'!r}. "
                f"MiniCPMOCRReader reads: {supported}."
            )

        import pymupdf

        document = pymupdf.open(path)
        try:
            matrix = pymupdf.Matrix(self.pdf_zoom, self.pdf_zoom)
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                yield (
                    index + 1,
                    Image.frombytes(
                        "RGB", (pixmap.width, pixmap.height), pixmap.samples
                    ),
                )
        finally:
            document.close()

    def _recognize_page(self, image: Image.Image) -> str:
        self._ensure_runtime()
        from minicpmvocr import assemble_markdown, convert_otsl_to_html, get_element_cfg
        from minicpmvocr.postprocess.markdown import postprocess_content

        prompt_map = self._prompt_map or {}
        detections = self._layout.detect([image])[0]
        blocks = self._layout.crop(image, detections)
        ignore = set(prompt_map.get("markdown_assembly", {}).get("ignore_labels", []))
        rendered: List[Dict[str, Any]] = []
        for order, block in enumerate(blocks):
            label = str(block.get("label", ""))
            if label in ignore or block.get("skip_vlm"):
                continue
            elem_cfg = get_element_cfg(prompt_map, label)
            if elem_cfg is None:
                continue
            crop = _block_image(image, block)
            if crop is None:
                logger.warning(
                    "Skipping %s block with an empty crop", label or "unknown"
                )
                continue
            prediction = self._engine.recognize(
                _resize_to_min_side(crop, elem_cfg.get("resize_min_side")),
                prompt=elem_cfg["prompt"],
                max_new_tokens=elem_cfg["max_new_tokens"],
                slice_nums=elem_cfg["slice_nums"],
            )
            finish_reason = prediction.get("finish_reason")
            if finish_reason not in (None, "", "stop"):
                logger.warning(
                    "Skipping truncated %s block: finish_reason=%s",
                    label,
                    finish_reason,
                )
                continue
            content = postprocess_content(
                prediction.get("text") or "",
                elem_cfg["output_format"],
                label=label,
            )
            if label in _TABLE_LABELS and _looks_like_table_markup(content):
                html = convert_otsl_to_html(content)
                if html:
                    content = html
            content = content.strip()
            if not content:
                continue
            rendered.append(
                {
                    "label": label,
                    "order": block.get("order", order),
                    "content": content,
                    "bbox": block.get("bbox"),
                }
            )
        return assemble_markdown(rendered, prompt_map).strip()

    def load_data(
        self,
        file_path: Union[str, Path],
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load one PDF or image into per-page documents.

        Args:
            file_path: Local PDF or image path.
            extra_info: Metadata merged into every document. Page and file
                fields from this reader take precedence.

        Returns:
            One document per non-empty page. ``metadata`` includes ``page``
            (1-based), ``file_name``, ``file_path``, and ``source``.

        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Expected a PDF or image file, got: {path}")

        documents: List[Document] = []
        with self._lock:
            for page_number, image in self._iter_pages(path):
                text = self._recognize_page(image)
                if not text:
                    continue
                metadata: Dict[str, Any] = dict(extra_info or {})
                metadata.update(
                    {
                        "file_name": path.name,
                        "file_path": str(path),
                        "page": page_number,
                        "source": str(path),
                    }
                )
                documents.append(Document(text=text, metadata=metadata))
        return documents
