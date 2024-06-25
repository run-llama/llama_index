import enum
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class TogglOutFormat(enum.Enum):
    json = "json"
    markdown = "markdown"


class TogglTrackItem(BaseModel):
    id: int
    pid: int
    tid: Optional[int]
    uid: int
    description: str
    start: datetime
    end: datetime
    updated: datetime
    dur: int
    user: str
    use_stop: bool
    client: Optional[str]
    project: str
    project_color: str
    project_hex_color: str
    task: Optional[str]
    is_billable: bool
    cur: Optional[str]
    tags: List[str]
