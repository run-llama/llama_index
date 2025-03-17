"""Test components."""

from typing import Any, List, Sequence, Dict

import pytest
from llama_index.core.base.base_selector import (
    BaseSelector,
    MultiSelection,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.query_pipeline.components.argpacks import (
    ArgPackComponent,
    KwargPackComponent,
)
from llama_index.core.query_pipeline.components.function import FnComponent
from llama_index.core.query_pipeline.components.input import InputComponent
from llama_index.core.query_pipeline.components.stateful import StatefulFnComponent
from llama_index.core.query_pipeline.components.loop import LoopComponent
from llama_index.core.query_pipeline.components.router import (
    RouterComponent,
    SelectorComponent,
)
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata


def foo_fn(a: int, b: int = 1, c: int = 2) -> int:
    """Foo function."""
    return a + b + c


def bar_fn(a: Any, b: Any) -> str:
    """Bar function."""
    return str(a) + ":" + str(b)


def sum_fn(a: List[int]) -> int:
    """Mock list function."""
    return sum(a)


def test_fn_components() -> None:
    """Test components."""
    foo_c = FnComponent(fn=foo_fn)
    assert foo_c.run_component(a=1) == {"output": 4}
    assert foo_c.run_component(a=1, b=100) == {"output": 103}
    foo_c = FnComponent(fn=foo_fn, output_key="foo")
    assert foo_c.run_component(a=1, b=100, c=1000) == {"foo": 1101}

    # try no positional args
    with pytest.raises(ValueError):
        foo_c.run_component(b=100, c=1000)

    # try bar
    bar_c = FnComponent(fn=bar_fn)
    assert bar_c.run_component(a="hello", b="world") == {"output": "hello:world"}
    # try one positional arg
    with pytest.raises(ValueError):
        bar_c.run_component(a="hello")
    # try extra kwargs
    with pytest.raises(ValueError):
        bar_c.run_component(a="hello", b="world", c="foo")


def test_fn_pipeline() -> None:
    """Test pipeline with function components."""
    p = QueryPipeline(chain=[FnComponent(fn=foo_fn), FnComponent(fn=foo_fn)])
    output = p.run(a=1)
    assert output == 7

    p2 = QueryPipeline()
    p2.add_modules(
        {"input": InputComponent(), "foo1": p, "foo2": p, "bar": FnComponent(fn=bar_fn)}
    )

    # draw links
    p2.add_link("input", "foo1", src_key="a")
    p2.add_link("input", "foo2", src_key="a")
    p2.add_link("foo1", "bar", dest_key="a")
    p2.add_link("foo2", "bar", dest_key="b")
    output = p2.run(a=1)
    assert output == "7:7"


def test_arg_component() -> None:
    """Test arg component."""
    arg_c = ArgPackComponent()
    assert arg_c.run_component(a=1, b=2) == {"output": [1, 2]}

    sum_c = FnComponent(fn=sum_fn)

    p = QueryPipeline(chain=[arg_c, sum_c])
    assert p.run(a=1, b=2) == 3


def test_kwarg_component() -> None:
    """Test kwarg component."""
    arg_c = KwargPackComponent()
    assert arg_c.run_component(a=1, b=2) == {"output": {"a": 1, "b": 2}}

    def convert_fn(d: dict) -> list:
        """Convert."""
        return list(d.values())

    convert_c = FnComponent(fn=convert_fn)
    sum_c = FnComponent(fn=sum_fn)

    p = QueryPipeline(chain=[arg_c, convert_c, sum_c])
    assert p.run(tmp=3, tmp2=2) == 5


class MockSelector(BaseSelector):
    """Mock selector."""

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        """Select."""
        return MultiSelection(
            selections=[SingleSelection(index=len(choices) - 1, reason="foo")]
        )

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        return self._select(choices, query)

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        """Update prompts."""


def test_selector_component() -> None:
    """Test selector component."""

    def bar1_fn(a: Any) -> str:
        """Bar function."""
        return str(a) + ":bar1"

    def bar2_fn(a: Any) -> str:
        """Bar function."""
        return str(a) + ":bar2"

    selector = MockSelector()
    router = RouterComponent(
        selector=selector,
        choices=["foo", "bar"],
        components=[FnComponent(fn=bar1_fn), FnComponent(fn=bar2_fn)],
    )
    assert router.run_component(query="hello") == {"output": "hello:bar2"}

    selector_c = SelectorComponent(selector=selector)
    output = selector_c.run_component(query="hello", choices=["t1", "t2"])
    assert output["output"][0] == SingleSelection(index=1, reason="foo")


def stateful_foo_fn(state: Dict[str, Any], a: int, b: int = 2) -> Dict[str, Any]:
    """Foo function."""
    old = state.get("prev", 0)
    new = old + a + b
    state["prev"] = new
    return new


def test_stateful_fn_pipeline() -> None:
    """Test pipeline with function components."""
    p = QueryPipeline()
    p.add_modules(
        {
            "m1": StatefulFnComponent(fn=stateful_foo_fn),
            "m2": StatefulFnComponent(fn=stateful_foo_fn),
        }
    )
    p.add_link("m1", "m2", src_key="output", dest_key="a")
    output = p.run(a=1, b=2)
    assert output == 8
    p.reset_state()
    output = p.run(a=1, b=2)
    assert output == 8

    # try one iteration
    p.reset_state()
    loop_component = LoopComponent(
        pipeline=p,
        should_exit_fn=lambda x: x["output"] > 10,
        # add_output_to_input_fn=lambda cur_input, output: {"a": output},
        max_iterations=1,
    )
    output = loop_component.run_component(a=1, b=2)
    assert output["output"] == 8

    # try two iterations
    p.reset_state()
    # loop 1: 0 + 1 + 2 = 3, 3 + 3 + 2 = 8
    # loop 2: 8 + 8 + 2 = 18, 18 + 18 + 2 = 38
    loop_component = LoopComponent(
        pipeline=p,
        should_exit_fn=lambda x: x["output"] > 10,
        add_output_to_input_fn=lambda cur_input, output: {"a": output["output"]},
        max_iterations=5,
    )
    assert loop_component.run_component(a=1, b=2)["output"] == 38

    # test loop component
