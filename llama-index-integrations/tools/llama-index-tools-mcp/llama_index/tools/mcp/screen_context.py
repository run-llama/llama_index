"""将 ScreenContext 的本地屏幕历史转换为 LlamaIndex 文档。"""

import asyncio
import json
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from llama_index.tools.mcp.client import BasicMCPClient


class ScreenContextReader(BaseReader):
    """
    通过 MCP 读取 ScreenContext 的本地屏幕历史。

    ScreenContext 会继续执行它自己的访问控制、排除策略和审计。这个 reader
    只做本地查询词匹配和时间范围筛选，不会把屏幕内容当作指令。

    Args:
        client: 已配置好的 ScreenContext MCP 客户端。客户端应连接到使用
            ``standard`` 或 ``full`` profile 启动的 ScreenContext server。

    """

    def __init__(self, client: BasicMCPClient) -> None:
        self.client = client

    async def aload_data(
        self,
        query: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 20,
        timezone_name: str = "UTC",
    ) -> list[Document]:
        """
        异步读取指定时间范围内包含查询词的屏幕观察记录。

        Args:
            query: 不区分大小写的字面查询词。
            start_time: 查询范围的起始时间，包含该时刻。
            end_time: 查询范围的结束时间，不包含该时刻；省略时使用当前时间。
            limit: 最多返回的文档数量。
            timezone_name: ScreenContext 使用的 IANA 时区名称。

        Returns:
            按观察时间升序排列的文档列表。

        Raises:
            ValueError: 查询参数、时间范围或 MCP 响应格式无效时抛出。

        """
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must contain 1–500 characters")
        query = query.strip()
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("limit must be between 1 and 100")

        try:
            zone = ZoneInfo(timezone_name)
        except (TypeError, ValueError):
            raise ValueError("timezone_name must be a valid IANA timezone") from None

        start = self._in_zone(start_time, zone, "start_time")
        end = self._in_zone(end_time or datetime.now(zone), zone, "end_time")
        if end <= start:
            raise ValueError("end_time must be after start_time")

        documents: list[Document] = []
        current = start.date()
        last_date = end.date()
        while current <= last_date and len(documents) < limit:
            cursor: Optional[str] = None
            seen_cursors: set[str] = set()
            while len(documents) < limit:
                arguments: dict[str, Any] = {
                    "date": current.isoformat(),
                    "timezone": timezone_name,
                    "purpose": "report",
                    "limit": 10,
                }
                if cursor is not None:
                    if cursor in seen_cursors:
                        raise ValueError("MCP returned a repeated pagination cursor")
                    seen_cursors.add(cursor)
                    arguments["cursor"] = cursor

                payload = await self._call_day_material(arguments)
                records = payload.get("records", [])
                if not isinstance(records, list):
                    raise ValueError("MCP response records must be a list")

                for record in records:
                    document = self._to_document(record, query, start, end, zone)
                    if document is not None:
                        documents.append(document)
                        if len(documents) == limit:
                            break

                next_cursor = payload.get("next_cursor")
                if len(documents) >= limit or not next_cursor:
                    break
                if not isinstance(next_cursor, str):
                    raise ValueError("MCP response next_cursor must be a string")
                cursor = next_cursor

            current = date.fromordinal(current.toordinal() + 1)

        documents.sort(key=lambda document: document.metadata["timestamp_unix"])
        return documents[:limit]

    def load_data(
        self,
        query: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 20,
        timezone_name: str = "UTC",
    ) -> list[Document]:
        """同步读取屏幕历史；异步环境请使用 ``aload_data``。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aload_data(query, start_time, end_time, limit, timezone_name)
            )
        raise RuntimeError(
            "Use await reader.aload_data(...) inside an async event loop"
        )

    async def _call_day_material(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.client.call_tool("get_day_material", arguments)
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise RuntimeError("ScreenContext MCP get_day_material failed")
        payload = self._payload(result)
        if not isinstance(payload, dict):
            raise ValueError("MCP response must be an object")
        return payload

    @staticmethod
    def _in_zone(value: datetime, zone: ZoneInfo, name: str) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError(f"{name} must be a datetime")
        if value.tzinfo is None:
            return value.replace(tzinfo=zone)
        return value.astimezone(zone)

    @staticmethod
    def _payload(result: Any) -> Any:
        if isinstance(result, Mapping):
            return result

        for attribute in ("structuredContent", "structured_content"):
            structured = getattr(result, attribute, None)
            if isinstance(structured, Mapping):
                return structured

        for block in getattr(result, "content", []):
            text = (
                block.get("text")
                if isinstance(block, Mapping)
                else getattr(block, "text", None)
            )
            if not isinstance(text, str):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                return payload

        raise ValueError("MCP response did not contain a JSON object")

    @staticmethod
    def _to_document(
        record: Any,
        query: str,
        start: datetime,
        end: datetime,
        zone: ZoneInfo,
    ) -> Optional[Document]:
        if not isinstance(record, Mapping):
            raise ValueError("MCP response records must contain objects")
        text = record.get("text")
        timestamp = record.get("observed_at")
        if not isinstance(text, str) or not text.strip():
            return None
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError("MCP record observed_at must be a number")
        observed_at = datetime.fromtimestamp(timestamp, zone)
        if not start <= observed_at < end or query.casefold() not in text.casefold():
            return None

        metadata = {
            "source": "observed_screen",
            "trust": "untrusted",
            "frame_id": record.get("frame_id"),
            "timestamp": observed_at.isoformat(),
            "timestamp_unix": timestamp,
            "app": record.get("app", ""),
            "app_bundle": record.get("app_bundle", ""),
            "window_title": record.get("title", ""),
            "domains": record.get("domains", []),
        }
        return Document(text=text, metadata=metadata)
