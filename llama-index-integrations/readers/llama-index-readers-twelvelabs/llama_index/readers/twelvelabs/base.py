"""TwelveLabs video reader — Pegasus on-the-fly analysis into Documents."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

API_BASE = "https://api.twelvelabs.io/v1.3"
DEFAULT_MODEL = "pegasus1.5"
DEFAULT_PROMPT = (
    "Describe this video in detail. Include what happens visually, what is said "
    "(transcribe the audio), any on-screen text, and the overall purpose, with "
    "[MM:SS] timestamps."
)
_POLL_INTERVAL_SECONDS = 3.0
_ASSET_TIMEOUT_SECONDS = 900
_ANALYZE_TIMEOUT_SECONDS = 1800
# Direct asset upload is capped at 200 MB; larger local files must be hosted and
# passed as a URL instead.
_MAX_DIRECT_UPLOAD_BYTES = 200 * 1024 * 1024


class TwelveLabsVideoReader(BaseReader):
    """
    Analyze videos with TwelveLabs Pegasus and load the result as Documents.

    Pegasus performs on-the-fly video-language analysis (visuals plus its own
    audio ASR) and returns text — so a single video becomes one ``Document``
    whose text is the model's analysis (e.g. a description + transcript), with no
    frame extraction and no separate transcription step.

    To use it, set the ``TWELVELABS_API_KEY`` environment variable (or pass
    ``api_key``). Get a key at https://playground.twelvelabs.io.

    Args:
        api_key: TwelveLabs API key. Falls back to ``TWELVELABS_API_KEY``.
        model: Pegasus model name (``pegasus1.5`` or ``pegasus1.2``).
        prompt: Default analysis prompt. Overridable per ``load_data`` call.
        temperature: Sampling temperature (0-1).
        max_tokens: Max output tokens per analysis.

    Example:
        >>> reader = TwelveLabsVideoReader()
        >>> docs = reader.load_data(
        ...     video_url="https://example.com/video.mp4",
        ...     prompt="Summarize the key moments.",
        ... )

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        prompt: str = DEFAULT_PROMPT,
        temperature: float = 0.2,
        max_tokens: int = 16384,
    ) -> None:
        key = api_key or os.environ.get("TWELVELABS_API_KEY")
        if not key:
            raise ValueError(
                "A TwelveLabs API key is required. Pass api_key=... or set the "
                "TWELVELABS_API_KEY environment variable."
            )
        self._api_key = key
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def load_data(
        self,
        video_url: Optional[str] = None,
        video_file: Optional[str] = None,
        asset_id: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> List[Document]:
        """
        Analyze one video and return it as a single-element list of Documents.

        Exactly one of ``video_url`` (a publicly accessible direct video URL),
        ``video_file`` (a local file path, <= 200 MB), or ``asset_id`` (an asset
        already uploaded to TwelveLabs) must be provided.

        Args:
            video_url: Publicly accessible direct video URL.
            video_file: Local video file path.
            asset_id: Existing TwelveLabs asset id.
            prompt: Analysis prompt for this call (defaults to ``self.prompt``).

        """
        provided = [s for s in (video_url, video_file, asset_id) if s]
        if len(provided) != 1:
            raise ValueError(
                "Provide exactly one of video_url, video_file, or asset_id."
            )

        resolved_asset = asset_id or self._resolve_asset(video_url, video_file)
        result = self._analyze(resolved_asset, prompt or self.prompt)

        metadata: Dict[str, Any] = {
            "source": video_url or video_file or f"asset:{resolved_asset}",
            "asset_id": resolved_asset,
            "model": self.model,
            "task_id": result.get("task_id"),
            "provider": "twelvelabs",
        }
        return [Document(text=result["text"], metadata=metadata)]

    # -- internal: TwelveLabs REST helpers ---------------------------------- #
    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": self._api_key}

    def _resolve_asset(
        self, video_url: Optional[str], video_file: Optional[str]
    ) -> str:
        if video_url:
            # `(None, value)` tuples force multipart/form-data (TwelveLabs rejects
            # urlencoded), without attaching a file.
            response = requests.post(
                f"{API_BASE}/assets",
                headers=self._headers(),
                files={"method": (None, "url"), "url": (None, video_url)},
                timeout=120,
            )
        else:
            path = Path(str(video_file)).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            if path.stat().st_size > _MAX_DIRECT_UPLOAD_BYTES:
                raise ValueError(
                    f"{path.name} exceeds the 200 MB direct-upload limit; host it "
                    "and pass video_url instead."
                )
            with path.open("rb") as handle:
                response = requests.post(
                    f"{API_BASE}/assets",
                    headers=self._headers(),
                    data={"method": "direct"},
                    files={"file": (path.name, handle)},
                    timeout=600,
                )
        self._raise_for_status(response, "asset upload")
        payload = response.json()
        asset_id = payload.get("_id") or payload.get("id")
        if not asset_id:
            raise ValueError(f"TwelveLabs asset upload returned no id: {payload}")
        if str(payload.get("status", "")).lower() != "ready":
            self._await_asset(asset_id)
        return asset_id

    def _await_asset(self, asset_id: str) -> None:
        deadline = time.monotonic() + _ASSET_TIMEOUT_SECONDS
        while True:
            response = requests.get(
                f"{API_BASE}/assets/{asset_id}", headers=self._headers(), timeout=60
            )
            self._raise_for_status(response, "asset status")
            status = str(response.json().get("status", "")).lower()
            if status == "ready":
                return
            if status == "failed":
                raise RuntimeError(f"TwelveLabs asset {asset_id} processing failed")
            if time.monotonic() > deadline:
                raise TimeoutError(f"TwelveLabs asset {asset_id} not ready in time")
            time.sleep(_POLL_INTERVAL_SECONDS)

    def _analyze(self, asset_id: str, prompt: str) -> Dict[str, Any]:
        body = {
            "video": {"type": "asset_id", "asset_id": asset_id},
            "model_name": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(
            f"{API_BASE}/analyze/tasks",
            headers=self._headers(),
            json=body,
            timeout=120,
        )
        self._raise_for_status(response, "analyze task")
        created = response.json()
        task_id = created.get("task_id") or created.get("_id") or created.get("id")
        if not task_id:
            raise ValueError(f"TwelveLabs analyze task returned no id: {created}")

        deadline = time.monotonic() + _ANALYZE_TIMEOUT_SECONDS
        while True:
            poll = requests.get(
                f"{API_BASE}/analyze/tasks/{task_id}",
                headers=self._headers(),
                timeout=60,
            )
            self._raise_for_status(poll, "analyze status")
            info = poll.json()
            status = str(info.get("status", "")).lower()
            if status == "ready":
                text = _extract_text(info.get("result"))
                if not text:
                    raise RuntimeError(
                        f"TwelveLabs task {task_id} ready but produced no text"
                    )
                return {"text": text, "task_id": task_id}
            if status == "failed":
                raise RuntimeError(f"TwelveLabs analyze task {task_id} failed: {info}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"TwelveLabs analyze task {task_id} not ready in time"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)

    @staticmethod
    def _raise_for_status(response: requests.Response, what: str) -> None:
        if not response.ok:
            raise RuntimeError(
                f"TwelveLabs {what} failed: HTTP {response.status_code} "
                f"{response.text[:300]}"
            )


def _extract_text(result: Any) -> str:
    """Pull the generated text out of an analyze task result (shape-tolerant)."""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("data", "text", "analysis", "summary", "generated_text", "output"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""
