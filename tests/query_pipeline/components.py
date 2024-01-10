"""Test components."""
from llama_index.core.query_pipeline.components import FnComponent
import pytest


def foo_fn(a: int, b: int = 1, c: int = 2) -> int:
    """Foo function."""
    return a + b + c


def bar_fn(a: str, b: str) -> str:
    """Bar function."""
    return a + ":" + b


def test_components() -> None:
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
        

    
