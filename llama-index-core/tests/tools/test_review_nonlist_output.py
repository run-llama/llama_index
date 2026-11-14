import asyncio

import numpy as np
import pytest

from llama_index.core.tools.function_tool import FunctionTool


@pytest.mark.parametrize("asynchronous", [False, True])
def test_function_tool_numpy_output(asynchronous):
    values = np.array([1, 2])

    def get_values():
        return values

    tool = FunctionTool.from_defaults(get_values)
    result = asyncio.run(tool.acall()) if asynchronous else tool.call()
    assert result.content == str(values)
    assert result.raw_output is values


def test_function_tool_empty_list_still_serializes():
    tool = FunctionTool.from_defaults(lambda: [])
    assert tool.call().content == "[]"
