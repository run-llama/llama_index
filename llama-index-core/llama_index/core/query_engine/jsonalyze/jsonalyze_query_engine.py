"""
JSONalyze Query Engine.

DEPRECATED AND REMOVED: This class has been removed due to security concerns.

WARNING: The JSONalyzeQueryEngine previously executed LLM-generated SQL prompts
with SQLite, which could lead to arbitrary file creation and other security issues.
Like PandasQueryEngine, sandboxing LLM-generated code execution proved inadequate
without external isolation.

For safe JSON/SQL operations, use explicit query builders with parameterized
queries and proper input validation, or run untrusted code in isolated sandbox
environments (Docker, gVisor, etc.).

"""

from typing import Any


class JSONalyzeQueryEngine:
    """
    JSONalyze query engine.

    DEPRECATED AND REMOVED: This class has been removed due to security concerns.
    See module docstring for details and alternatives.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise DeprecationWarning(
            "JSONalyzeQueryEngine has been removed due to security vulnerabilities.\n"
            "LLM-generated SQL execution could not be adequately sandboxed, leading to\n"
            "arbitrary file creation and other security risks.\n\n"
            "Alternatives:\n"
            "1. Use explicit SQL query builders with parameterized queries\n"
            "2. Validate and sanitize all LLM-generated queries before execution\n"
            "3. Run untrusted code in external sandboxed environments (Docker, gVisor, etc.)\n\n"
            "Do not attempt to recreate this functionality without proper external sandboxing."
        )
