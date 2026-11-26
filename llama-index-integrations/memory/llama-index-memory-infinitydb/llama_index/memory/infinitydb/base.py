"""InfinityDB Drop-in Memory Adapter for LlamaIndex."""

from __future__ import annotations

from typing import Any, List, Optional

from infinitydb import InfinityDB


class InfinityDBLlamaMemory:
    """Drop-in memory interface compatible with LlamaIndex BaseMemory specification.

    Provides sub-millisecond, local-first neurosymbolic memory for LlamaIndex
    agents, chat engines, and query pipelines with zero LLM extraction cost.
    """

    def __init__(
        self,
        db: Optional[Any] = None,
        db_path: str = "llama_memory.db",
        user_id: str = "default_user",
        session_id: Optional[str] = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> None:
        if isinstance(db, str):
            self.db = InfinityDB(db)
        elif db is not None:
            self.db = db
        else:
            self.db = InfinityDB(kwargs.get("db_path", db_path))
        self.user_id = user_id
        self.session_id = session_id
        self.limit = limit

    def get(self, input_text: Optional[str] = None, **kwargs: Any) -> str:
        """Retrieve formatted memory context for LlamaIndex prompt injection."""
        if not input_text:
            persona = self.db.get_persona(user_id=self.user_id)
            if not persona:
                return ""
            return "User Profile:\n" + "\n".join(f"- {k}: {v}" for k, v in persona.items())

        res = self.db.recall(query=input_text, n=self.limit, user_id=self.user_id)
        facts = [item["fact"] for item in res.get("results", [])]
        if not facts:
            return ""
        return "Relevant Long-Term Memories:\n" + "\n".join(f"- {f}" for f in facts)

    def put(self, message: str, role: str = "user", **kwargs: Any) -> str:
        """Store dialogue turn into L0-L6 Brain pipeline with zero LLM token costs."""
        content = f"{role.capitalize()}: {message}"
        return self.db.add(
            conversation=content,
            user_id=self.user_id,
            session_id=self.session_id,
            importance=0.9,
        )

    def reset(self) -> None:
        """Clear memories for active user."""
        self.db.delete_user(self.user_id)
