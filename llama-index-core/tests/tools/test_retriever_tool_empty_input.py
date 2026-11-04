from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.tools import RetrieverTool


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_retriever_tool_rejects_missing_input(use_async: bool) -> None:
    retriever = MagicMock(spec=BaseRetriever)
    retriever.retrieve.return_value = []
    retriever.aretrieve = AsyncMock(return_value=[])
    tool = RetrieverTool.from_defaults(retriever=retriever)

    with pytest.raises(ValueError, match="Cannot call query engine without inputs"):
        if use_async:
            await tool.acall()
        else:
            tool.call()

    retriever.retrieve.assert_not_called()
    retriever.aretrieve.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("args", "kwargs", "query"),
    [((0,), {}, "0\n\n"), (("",), {}, "\n\n"), ((), {"limit": 0}, "\nlimit is 0\n")],
)
async def test_retriever_tool_accepts_supplied_inputs(
    use_async, args, kwargs, query
) -> None:
    retriever = MagicMock(spec=BaseRetriever)
    retriever.retrieve.return_value = []
    retriever.aretrieve = AsyncMock(return_value=[])
    tool = RetrieverTool.from_defaults(retriever=retriever)

    result = (
        await tool.acall(*args, **kwargs) if use_async else tool.call(*args, **kwargs)
    )

    assert result.raw_input["input"] == query
    assert result.raw_output == []
