"""Compatibility shim — implementation lives in multi_modal_context.py."""

from llama_index.core.chat_engine.multi_modal_context import (  # noqa: F401
    DEFAULT_CONTEXT_TEMPLATE,
    MultiModalContextChatEngine,
)

__all__ = ["MultiModalContextChatEngine", "DEFAULT_CONTEXT_TEMPLATE"]
