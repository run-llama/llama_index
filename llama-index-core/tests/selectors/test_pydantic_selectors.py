from typing import Any, Type

import pytest

from llama_index.core.base.base_selector import MultiSelection, SingleSelection
from llama_index.core.selectors.pydantic_selectors import PydanticMultiSelector
from llama_index.core.types import BasePydanticProgram


class AsyncOnlyMultiSelectionProgram(BasePydanticProgram[MultiSelection]):
    @property
    def output_cls(self) -> Type[MultiSelection]:
        return MultiSelection

    def __call__(self, *args: Any, **kwargs: Any) -> MultiSelection:
        raise AssertionError("The synchronous program should not be called")

    async def acall(self, *args: Any, **kwargs: Any) -> MultiSelection:
        return MultiSelection(
            selections=[SingleSelection(index=1, reason="most relevant")]
        )


@pytest.mark.asyncio
async def test_pydantic_multi_selector_uses_async_program() -> None:
    selector = PydanticMultiSelector(
        selector_program=AsyncOnlyMultiSelectionProgram(),
        max_outputs=1,
    )

    result = await selector.aselect(
        choices=["apple", "pear"],
        query="Which fruit is most relevant?",
    )

    assert result.inds == [0]
