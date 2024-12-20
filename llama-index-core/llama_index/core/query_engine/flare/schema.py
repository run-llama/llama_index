"""FLARE schema."""

from dataclasses import dataclass


@dataclass
class QueryTask:
    """Query task."""

    query_str: str
    start_idx: int
    end_idx: int
