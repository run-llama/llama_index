import asyncio
import datetime
from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from llama_index.readers.toggl.dto import TogglTrackItem, TogglOutFormat


class TogglReader(BaseReader):
    def __init__(
        self, api_token: str, user_agent: str = "llama_index_toggl_reader"
    ) -> None:
        """Initialize with parameters."""
        super().__init__()
        self.api_token = api_token
        self.user_agent = user_agent
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def load_data(
        self,
        workspace_id: str,
        project_id: str,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = datetime.datetime.now(),
        out_format: TogglOutFormat = TogglOutFormat.json,
    ) -> List[Document]:
        """
        Load data from Toggl.

        Args:
            workspace_id (str): The workspace ID.
            project_id (str): The project ID.
            start_date (Optional[datetime.datetime]): The start date.
            end_date (Optional[datetime.datetime]): The end date.
            out_format (TogglOutFormat): The output format.

        """
        return self.loop.run_until_complete(
            self.aload_data(workspace_id, project_id, start_date, end_date, out_format)
        )

    async def aload_data(
        self,
        workspace_id: str,
        project_id: str,
        start_date: Optional[datetime.datetime],
        end_date: Optional[datetime.datetime],
        out_format: TogglOutFormat,
    ) -> List[Document]:
        """Load time entries from Toggl."""
        from toggl.api_client import TogglClientApi

        client = TogglClientApi(
            {
                "token": self.api_token,
                "workspace_id": workspace_id,
                "user_agent": self.user_agent,
            }
        )
        project_times = client.get_project_times(project_id, start_date, end_date)
        raw_items = [
            TogglTrackItem.model_validate(raw_item)
            for raw_item in project_times["data"]
        ]
        items = []
        for item in raw_items:
            if out_format == TogglOutFormat.json:
                text = item.model_dump_json()
            elif out_format == TogglOutFormat.markdown:
                text = f"""# {item.description}
                    **Start:** {item.start:%Y-%m-%d %H:%M:%S%z}
                    **End:** {item.end:%Y-%m-%d %H:%M:%S%z}
                    **Duration:** {self.milliseconds_to_postgresql_interval(item.dur)}
                    **Tags:** {",".join(item.tags)}
                """
            doc = Document(text=text)
            doc.metadata = {**doc.metadata, **item.dict()}
            items.append(doc)
        return items

    def milliseconds_to_postgresql_interval(self, milliseconds):
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        interval = ""
        if days > 0:
            interval += f"{days}d"
        if hours > 0:
            interval += f"{hours}h"
        if minutes > 0:
            interval += f"{minutes}m"
        if seconds > 0 or milliseconds > 0:
            interval += f"{seconds}s"
        if milliseconds > 0:
            interval += f"{milliseconds}ms"

        return interval
