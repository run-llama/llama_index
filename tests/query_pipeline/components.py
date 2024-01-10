"""Test components."""
from typing import Any, List

import pytest
from llama_index.core.query_pipeline.components import (
    ArgPackComponent,
    FnComponent,
    InputComponent,
    KwargPackComponent,
)
from llama_index.query_pipeline.query import QueryPipeline


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
