import warnings

warnings.warn(
    "llama-index-packs-streamlit-chatbot is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.streamlit_chatbot.base import StreamlitChatPack

__all__ = ["StreamlitChatPack"]
