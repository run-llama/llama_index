"""Test components."""
from typing import Any

import pytest
from llama_index.core.query_pipeline.components import FnComponent, InputComponent
from llama_index.query_pipeline.query import QueryPipeline


def foo_fn(a: int, b: int = 1, c: int = 2) -> int:
    """Foo function."""
    return a + b + c


def bar_fn(a: Any, b: Any) -> str:
    """Bar function."""
    return str(a) + ":" + str(b)


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
