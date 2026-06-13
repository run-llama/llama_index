from unittest.mock import patch

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.memory.recollect import RecollectContext, RecollectMemory


def test_recollect_memory_from_config(tmp_path):
    context = {"user_id": "test_user"}
    config = {
        "data_dir": str(tmp_path / "recollect"),
        "extraction_enabled": False,
        "embedder": {"provider": "local", "dimensions": 64},
        "vector_store": {"provider": "sqlite", "embedding_dims": 64},
    }

    with patch("llama_index.memory.recollect.base.Memory") as MockMemory:
        mock_client = MockMemory.return_value
        mock_client.search.return_value = {
            "results": [{"memory": "User likes hiking", "score": 0.9}]
        }

        memory = RecollectMemory.from_config(
            context=context,
            config=config,
            search_msg_limit=3,
        )

        assert isinstance(memory, RecollectMemory)
        assert isinstance(memory.context, RecollectContext)
        assert memory.context.user_id == "test_user"
        assert memory._client == mock_client
        assert memory.search_msg_limit == 3


def test_recollect_memory_get_injects_system_message(tmp_path):
    config = {
        "data_dir": str(tmp_path / "recollect"),
        "extraction_enabled": False,
        "embedder": {"provider": "local", "dimensions": 64},
        "vector_store": {"provider": "sqlite", "embedding_dims": 64},
    }
    memory = RecollectMemory.from_config(context={"user_id": "bob"}, config=config)
    assert memory._client is not None
    memory._client.add("User lives in Cape Town", user_id="bob", infer=False)

    memory.primary_memory.put(ChatMessage(role=MessageRole.USER, content="Where do I live?"))
    messages = memory.get(input="Where do I live?")
    assert messages[0].role == MessageRole.SYSTEM
    assert "Cape Town" in (messages[0].content or "")