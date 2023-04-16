


from enum import Enum


class ResponseMode(str, Enum):
    """Response modes."""

    DEFAULT = "default"
    COMPACT = "compact"
    TREE_SUMMARIZE = "tree_summarize"
    NO_TEXT = "no_text"
