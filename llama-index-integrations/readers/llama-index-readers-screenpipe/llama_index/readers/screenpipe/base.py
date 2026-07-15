"""Screenpipe reader."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

SEARCH_URL_TMPL = "{base_url}/search"

VALID_CONTENT_TYPES = {
    "all",
    "ocr",
    "audio",
    "ui",
    "audio+ui",
    "ocr+ui",
    "audio+ocr",
}


class ScreenpipeReader(BasePydanticReader):
    """
    Screenpipe reader.

    Reads screen capture (OCR) and audio transcription data from a local
    Screenpipe instance via its REST API.

    See https://github.com/mediar-ai/screenpipe for details.

    Args:
        base_url (str): Base URL of the Screenpipe server.
            Defaults to ``http://localhost:3030``.

    """

    is_remote: bool = True
    base_url: str = "http://localhost:3030"

    @classmethod
    def class_name(cls) -> str:
        return "ScreenpipeReader"

    @staticmethod
    def _to_utc_isoformat(dt: datetime) -> str:
        """
        Convert a datetime to a UTC ISO 8601 string.

        Screenpipe requires UTC timestamps. Naive datetimes (no tzinfo) are
        assumed to be local time and converted to UTC.
        """
        if dt.tzinfo is None:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a search request against the Screenpipe API."""
        url = SEARCH_URL_TMPL.format(base_url=self.base_url.rstrip("/"))
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def load_data(
        self,
        content_type: str = "all",
        query: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        app_name: Optional[str] = None,
        window_name: Optional[str] = None,
        limit: int = 20,
    ) -> List[Document]:
        """
        Load data from Screenpipe.

        Args:
            content_type: Type of content to retrieve.
                One of ``"all"``, ``"ocr"``, ``"audio"``, ``"ui"``,
                ``"audio+ui"``, ``"ocr+ui"``, ``"audio+ocr"``.
            query: Optional search query for semantic filtering.
            start_time: Filter results after this timestamp.
            end_time: Filter results before this timestamp.
            app_name: Filter by application name.
            window_name: Filter by window name.
            limit: Maximum number of results to return.

        Returns:
            List of documents.

        """
        if content_type not in VALID_CONTENT_TYPES:
            raise ValueError(
                f"Invalid content_type '{content_type}'. "
                f"Must be one of: {sorted(VALID_CONTENT_TYPES)}"
            )

        params: Dict[str, Any] = {
            "content_type": content_type,
            "limit": limit,
        }
        if query is not None:
            params["q"] = query
        if start_time is not None:
            params["start_time"] = self._to_utc_isoformat(start_time)
        if end_time is not None:
            params["end_time"] = self._to_utc_isoformat(end_time)
        if app_name is not None:
            params["app_name"] = app_name
        if window_name is not None:
            params["window_name"] = window_name

        all_items: List[Dict[str, Any]] = []
        offset = 0

        while True:
            params["offset"] = offset
            data = self._search(params)
            items = data.get("data", [])
            if not items:
                break
            all_items.extend(items)
            if len(all_items) >= limit:
                all_items = all_items[:limit]
                break
            pagination = data.get("pagination", {})
            total = pagination.get("total", 0)
            offset += len(items)
            if offset >= total:
                break

        documents = []
        for item in all_items:
            doc = self._item_to_document(item)
            if doc is not None:
                documents.append(doc)

        return documents

    def _item_to_document(self, item: Dict[str, Any]) -> Optional[Document]:
        """Convert a Screenpipe search result item to a Document."""
        item_type = item.get("type", "")
        content = item.get("content", {})

        if item_type == "OCR":
            return self._ocr_to_document(content)
        elif item_type == "Audio":
            return self._audio_to_document(content)
        elif item_type == "UI":
            return self._ui_to_document(content)
        else:
            logger.warning("Unknown item type '%s', skipping.", item_type)
            return None

    def _ocr_to_document(self, content: Dict[str, Any]) -> Document:
        """Convert an OCR content item to a Document."""
        text = content.get("text", "")
        metadata: Dict[str, Any] = {
            "type": "ocr",
            "app_name": content.get("app_name", ""),
            "window_name": content.get("window_name", ""),
            "timestamp": content.get("timestamp", ""),
        }
        if content.get("file_path"):
            metadata["file_path"] = content["file_path"]
        if content.get("browser_url"):
            metadata["browser_url"] = content["browser_url"]
        return Document(text=text, metadata=metadata)

    def _audio_to_document(self, content: Dict[str, Any]) -> Document:
        """Convert an Audio content item to a Document."""
        text = content.get("transcription", "")
        metadata: Dict[str, Any] = {
            "type": "audio",
            "device_name": content.get("device_name", ""),
            "device_type": content.get("device_type", ""),
            "timestamp": content.get("timestamp", ""),
        }
        if content.get("file_path"):
            metadata["file_path"] = content["file_path"]
        speaker = content.get("speaker")
        if speaker:
            metadata["speaker_id"] = speaker.get("id")
            metadata["speaker_name"] = speaker.get("name")
        return Document(text=text, metadata=metadata)

    def _ui_to_document(self, content: Dict[str, Any]) -> Document:
        """Convert a UI content item to a Document."""
        text = content.get("text", "")
        metadata: Dict[str, Any] = {
            "type": "ui",
            "app_name": content.get("app_name", ""),
            "window_name": content.get("window_name", ""),
            "timestamp": content.get("timestamp", ""),
        }
        if content.get("browser_url"):
            metadata["browser_url"] = content["browser_url"]
        return Document(text=text, metadata=metadata)
