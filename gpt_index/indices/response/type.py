from enum import Enum


class ResponseMode(str, Enum):
    """Response modes."""

    REFINE = "refine"
    COMPACT = "compact"
    SIMPLE_SUMMARIZE = "simple_summarize"
    TREE_SUMMARIZE = "tree_summarize"
    GENERATION = "generation"
    NO_TEXT = "no_text"
