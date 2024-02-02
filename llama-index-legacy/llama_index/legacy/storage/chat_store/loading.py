from llama_index.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.simple_chat_store import SimpleChatStore

RECOGNIZED_CHAT_STORES = {
    SimpleChatStore.class_name(): SimpleChatStore,
}


def load_chat_store(data: dict) -> BaseChatStore:
    """Load a chat store from a dict."""
    chat_store_name = data.get("class_name", None)
    if chat_store_name is None:
        raise ValueError("ChatStore loading requires a class_name")

    if chat_store_name not in RECOGNIZED_CHAT_STORES:
        raise ValueError(f"Invalid ChatStore name: {chat_store_name}")

    return RECOGNIZED_CHAT_STORES[chat_store_name].from_dict(data)
