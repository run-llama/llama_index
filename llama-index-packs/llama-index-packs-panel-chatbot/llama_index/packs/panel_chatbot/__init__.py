import warnings

warnings.warn(
    "llama-index-packs-panel-chatbot is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.panel_chatbot.base import PanelChatPack

__all__ = ["PanelChatPack"]
