"""iLoveVideoEditor tool spec."""

import json
import os
import time
from typing import Any, Dict, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://api.ilovevideoeditor.com"
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


class ILoveVideoEditorTool(BaseToolSpec):
    """
    iLoveVideoEditor cloud video rendering tool.

    Renders MP4 videos from VideoJSON specifications using the
    iLoveVideoEditor API (https://ilovevideoeditor.com/docs/api-guide).

    Attributes:
        api_key: iLoveVideoEditor API key (env: ILOVEVIDEOEDITOR_API_KEY).
        base_url: API base URL (env: ILOVEVIDEOEDITOR_API_BASE).
        max_wait_seconds: Maximum time to poll for render completion.
        poll_interval_seconds: Interval between render status polls.

    """

    spec_functions = ["render_video", "get_render_status"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_wait_seconds: float = 300.0,
        poll_interval_seconds: float = 2.0,
    ) -> None:
        """
        Initialize the iLoveVideoEditor tool.

        Args:
            api_key: iLoveVideoEditor API key. Falls back to the
                ILOVEVIDEOEDITOR_API_KEY environment variable.
            base_url: API base URL. Falls back to the
                ILOVEVIDEOEDITOR_API_BASE environment variable, then to
                https://api.ilovevideoeditor.com.
            max_wait_seconds: Maximum time to poll for render completion.
            poll_interval_seconds: Interval between render status polls.

        Raises:
            ValueError: If no API key is provided or found in the environment.

        """
        self.api_key = api_key or os.environ.get("ILOVEVIDEOEDITOR_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing iLoveVideoEditor API key — pass api_key or set "
                "ILOVEVIDEOEDITOR_API_KEY"
            )
        self.base_url = (
            base_url or os.environ.get("ILOVEVIDEOEDITOR_API_BASE") or DEFAULT_BASE_URL
        )
        self.max_wait_seconds = max_wait_seconds
        self.poll_interval_seconds = poll_interval_seconds

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "user-agent": "llama-index-tools-ilovevideoeditor/0.1.0",
        }

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            json=payload,
            headers=self._headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize_result(state: dict) -> dict:
        """Reduce an API render object to the fields useful for an agent."""
        result = {
            "jobId": state.get("jobId") or state.get("id"),
            "status": state.get("status"),
        }
        progress = state.get("progress")
        if isinstance(progress, dict) and progress.get("percent") is not None:
            result["progress"] = progress["percent"]
        for key in ("downloadUrl", "error"):
            if state.get(key):
                result[key] = state[key]
        return result

    def render_video(
        self, video_json: Dict[str, Any], wait_for_completion: bool = True
    ) -> str:
        """
        Render an MP4 video from a VideoJSON specification.

        Args:
            video_json: VideoJSON object: {"name": str, "layers": [...]} where
                each layer has a "type" ("text", "image", "video", "audio",
                "captions", or "shape") and "settings" (startTime, duration,
                plus type-specific fields).
            wait_for_completion: When True (default), poll until the render
                finishes and return the download URL. When False, return the
                queued job ID immediately — use get_render_status to check on
                it later.

        Returns:
            A JSON string with jobId, status, downloadUrl (when completed) and
            error (when failed).

        Example:
            >>> tool = ILoveVideoEditorTool()
            >>> result = tool.render_video({
            ...     "name": "hello",
            ...     "layers": [
            ...         {
            ...             "type": "text",
            ...             "settings": {
            ...                 "startTime": 0,
            ...                 "duration": 3,
            ...                 "text": "Hello from LlamaIndex",
            ...             },
            ...         }
            ...     ],
            ... })
            >>> print(result)
            {"jobId": "425ba18a-...", "status": "completed", "downloadUrl": "https://..."}

        """
        try:
            queued = self._request("POST", "/v1/render", {"videoJSON": video_json})
            job_id = queued.get("jobId") or queued.get("id")
            if not wait_for_completion:
                return json.dumps(self._normalize_result(queued))

            deadline = time.monotonic() + self.max_wait_seconds
            state: dict = {}
            while time.monotonic() < deadline:
                state = self._request("GET", f"/v1/render/{job_id}")
                if state.get("status") in TERMINAL_STATUSES:
                    break
                time.sleep(self.poll_interval_seconds)
            else:
                return json.dumps(
                    {
                        "jobId": job_id,
                        "status": state.get("status", "unknown"),
                        "error": (
                            f"Render did not finish within "
                            f"{self.max_wait_seconds}s; "
                            f"check status later with the job ID."
                        ),
                    }
                )

            result = self._normalize_result(state)
            if state.get("status") == "completed" and not result.get("downloadUrl"):
                download = self._request("GET", f"/v1/render/{job_id}/download-url")
                url = download.get("downloadUrl") or download.get("url")
                if url:
                    result["downloadUrl"] = url
            return json.dumps(result)
        except Exception as e:
            return f"Error rendering video: {e}"

    def get_render_status(self, job_id: str) -> str:
        """
        Check the status of a previously queued render job.

        Args:
            job_id: The job ID returned by render_video.

        Returns:
            A JSON string with jobId, status, progress, downloadUrl (when
            completed) and error (when failed).

        """
        try:
            state = self._request("GET", f"/v1/render/{job_id}")
            return json.dumps(self._normalize_result(state))
        except Exception as e:
            return f"Error getting render status: {e}"
