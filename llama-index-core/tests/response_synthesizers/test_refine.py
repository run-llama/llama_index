from collections import OrderedDict
from typing import Any, Dict, Optional, Type, cast

import pytest
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.response_synthesizers import Refine
from llama_index.core.response_synthesizers.refine import StructuredRefineResponse
from llama_index.core.types import BasePydanticProgram


class MockRefineProgram(BasePydanticProgram):
    """
    Runs the query on the LLM as normal and always returns the answer with
    query_satisfied=True. In effect, doesn't do any answer filtering.
    """

    def __init__(self, input_to_query_satisfied: Dict[str, bool]):
        self._input_to_query_satisfied = input_to_query_satisfied

    @property
    def output_cls(self) -> Type[BaseModel]:
        return StructuredRefineResponse

    def __call__(
        self,
        *args: Any,
        context_str: Optional[str] = None,
        context_msg: Optional[str] = None,
        **kwargs: Any,
    ) -> StructuredRefineResponse:
        input_str = context_str or context_msg
        input_str = cast(str, input_str)
        query_satisfied = self._input_to_query_satisfied[input_str]
        return StructuredRefineResponse(
            answer=input_str, query_satisfied=query_satisfied
        )

    async def acall(
        self,
        *args: Any,
        context_str: Optional[str] = None,
        context_msg: Optional[str] = None,
        **kwargs: Any,
    ) -> StructuredRefineResponse:
        input_str = context_str or context_msg
        input_str = cast(str, input_str)
        query_satisfied = self._input_to_query_satisfied[input_str]
        return StructuredRefineResponse(
            answer=input_str, query_satisfied=query_satisfied
        )


@pytest.fixture()
def refine_instance() -> Refine:
    return Refine(
        streaming=False,
        verbose=True,
        structured_answer_filtering=True,
    )


def test_constructor_args() -> None:
    with pytest.raises(ValueError):
        # can't construct refine with both streaming and answer filtering
        Refine(
            streaming=True,
            structured_answer_filtering=True,
        )
    with pytest.raises(ValueError):
        # can't construct refine with a program factory but not answer filtering
        Refine(
            program_factory=lambda _: MockRefineProgram({}),
            structured_answer_filtering=False,
        )


@pytest.mark.asyncio
async def test_answer_filtering_one_answer() -> None:
    input_to_query_satisfied = OrderedDict(
        [
            ("input1", False),
            ("input2", True),
            ("input3", False),
        ]
    )

    def program_factory(*args: Any, **kwargs: Any) -> MockRefineProgram:
        return MockRefineProgram(input_to_query_satisfied)

    refine_instance = Refine(
        structured_answer_filtering=True,
        program_factory=program_factory,
    )
    res = await refine_instance.aget_response(
        "question", list(input_to_query_satisfied.keys())
    )
    assert res == "input2"


@pytest.mark.asyncio
async def test_answer_filtering_no_answers() -> None:
    input_to_query_satisfied = OrderedDict(
        [
            ("input1", False),
            ("input2", False),
            ("input3", False),
        ]
    )

    def program_factory(*args: Any, **kwargs: Any) -> MockRefineProgram:
        return MockRefineProgram(input_to_query_satisfied)

    refine_instance = Refine(
        structured_answer_filtering=True,
        program_factory=program_factory,
    )
    res = await refine_instance.aget_response(
        "question", list(input_to_query_satisfied.keys())
    )
    assert res == "Empty Response"
