import os
import pytest
from typing import List
from unittest.mock import MagicMock, patch, AsyncMock
import uuid

from llama_index.core.base.base_selector import (
    SelectorResult,
    SingleSelection,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import ToolMetadata
from llama_index.selectors.notdiamond.base import NotDiamondSelector, LLMSingleSelector

from notdiamond import LLMConfig


@pytest.fixture()
def session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture()
def choices() -> List[ToolMetadata]:
    return [
        ToolMetadata(
            name="vector_index", description="Great for asking questions about recipes."
        ),
        ToolMetadata(name="list_index", description="Great for summarizing recipes."),
    ]


@pytest.fixture()
def nd_selector(session_id):
    from notdiamond import NotDiamond

    os.environ["OPENAI_API_KEY"] = "test"
    os.environ["ANTHROPIC_API_KEY"] = "test"

    llm_configs = [
        LLMConfig(provider="openai", model="gpt-4o"),
    ]

    # mocking out model_select calls on client
    _client = MagicMock(stub=NotDiamond, api_key="test", llm_configs=llm_configs)
    _client.model_select.return_value = (session_id, llm_configs[0])

    async def aselect(*args, **kwargs):
        return (session_id, llm_configs[0])

    _client.amodel_select = aselect
    selector = NotDiamondSelector(client=_client)

    # monkeypatch the _select and _aselect methods on parent class of NDSelector
    LLMSingleSelector._select = MagicMock(
        return_value=SelectorResult(
            selections=[SingleSelection(index=0, reason="test")]
        )
    )
    LLMSingleSelector._aselect = AsyncMock(
        return_value=SelectorResult(
            selections=[SingleSelection(index=1, reason="test")]
        )
    )

    return selector


class TestNotDiamondSelector:
    @patch("llama_index.llms.openai.OpenAI")
    def test_select(self, openai_mock, nd_selector, choices, session_id):
        """_select should call openai, as mocked."""
        openai_mock.return_value = MagicMock()
        openai_mock.return_value.chat.return_value.message.content = "vector_index"
        query = "Please describe the llama_index framework in 280 characters or less."
        result = nd_selector._select(choices, QueryBundle(query_str=query))
        assert result.session_id == session_id
        assert str(result.llm) == "openai/gpt-4o"
        assert result.selections[0].index == 0
        assert openai_mock.is_called

    @pytest.mark.asyncio()
    @patch("llama_index.llms.openai.OpenAI")
    async def test_aselect(self, openai_mock, nd_selector, choices, session_id):
        """_aselect should call openai, as mocked."""
        openai_mock.return_value = MagicMock()
        openai_mock.return_value.chat.return_value.message.content = "vector_index"

        query = "How can I cook a vegan variant of deviled eggs?"
        result = await nd_selector._aselect(choices, QueryBundle(query_str=query))
        assert result.session_id == session_id
        assert str(result.llm) == "openai/gpt-4o"
        assert result.selections[0].index == 1
        assert openai_mock.is_called
