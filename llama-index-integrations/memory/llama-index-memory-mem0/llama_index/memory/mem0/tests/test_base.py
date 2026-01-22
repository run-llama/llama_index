from unittest.mock import MagicMock, patch

import pytest

from llama_index.memory.mem0.base import Mem0Memory, Mem0Context
from llama_index.core.memory import Memory as LlamaIndexMemory
from llama_index.core.base.llms.types import MessageRole


def test_mem0_search_merges_user_and_agent_results():
    """When both user_id and agent_id are provided, results should be merged."""

    with patch("llama_index.memory.mem0.base.MemoryClient") as MockClient:
        mock_client = MockClient.return_value

        mock_client.search.side_effect = [
            [{"content": "Anne likes biking"}],
            [{"content": "Peter hates stew"}],
        ]

        primary_memory = LlamaIndexMemory.from_defaults()
        primary_memory.get = MagicMock(return_value=[])
        primary_memory.get_all = MagicMock(return_value=[])

        context = Mem0Context(user_id="user123", agent_id="agent123")

        memory = Mem0Memory(
            primary_memory=primary_memory,
            context=context,
            client=mock_client,
        )

        messages = memory.get("What do we know about Anne and Peter?")

        assert messages
        assert messages[0].role == MessageRole.SYSTEM
        assert "Anne likes biking" in messages[0].content
        assert "Peter hates stew" in messages[0].content
        assert mock_client.search.call_count == 2
