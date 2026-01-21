from unittest.mock import MagicMock, patch

from llama_index.memory.mem0.base import Mem0Memory, Mem0Context
from llama_index.core.memory import Memory as LlamaIndexMemory
from llama_index.core.base.llms.types import MessageRole


def test_mem0_search_merges_user_and_agent_results():
    """
    When both user_id and agent_id are provided in context, Mem0Memory
    should perform separate searches and merge the results.
    """

    # Patch the MemoryClient where it is used
    with patch("llama_index.memory.mem0.base.MemoryClient") as MockClient:
        mock_client = MockClient.return_value

        # Two separate search results:
        # first call -> user_id search
        # second call -> agent_id search
        mock_client.search.side_effect = [
            [{"content": "Anne likes biking"}],
            [{"content": "Peter hates stew"}],
        ]

        # Mock primary memory (no chat history)
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

        # One system message should be added at the top
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[0].content is not None

        # Both memories should be present in merged system message
        assert "Anne likes biking" in messages[0].content
        assert "Peter hates stew" in messages[0].content

        # Ensure Mem0 search was called twice (user + agent)
        assert mock_client.search.call_count == 2
