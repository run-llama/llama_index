"""Query pipeline."""

from typing import Any, Dict

import pytest
from llama_index.core.query_pipeline.components.function import FnComponent
from llama_index.core.query_pipeline.components.input import InputComponent
from llama_index.core.base.query_pipeline.query import (
    ChainableMixin,
    InputKeys,
    Link,
    OutputKeys,
    QueryComponent,
)
from llama_index.core.query_pipeline.components.input import InputComponent
from llama_index.core.query_pipeline.query import QueryPipeline


class QueryComponent1(QueryComponent):
    """
    Query component 1.

    Adds two numbers together.

    """

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "input1" not in input:
            raise ValueError("input1 not in input")
        if "input2" not in input:
            raise ValueError("input2 not in input")
        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        return {"output": kwargs["input1"] + kwargs["input2"]}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"input1", "input2"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


class QueryComponent2(QueryComponent):
    """
    Query component 1.

    Joins two strings together with ':'

    """

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "input1" not in input:
            raise ValueError("input1 not in input")
        if "input2" not in input:
            raise ValueError("input2 not in input")
        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        return {"output": f"{kwargs['input1']}:{kwargs['input2']}"}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"input1", "input2"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


class QueryComponent3(QueryComponent):
    """
    Query component 3.

    Takes one input and doubles it.

    """

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "input" not in input:
            raise ValueError("input not in input")
        return input

    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""
        return {"output": kwargs["input"] + kwargs["input"]}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"input"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


class Chainable2(ChainableMixin):
    """Chainable mixin."""

    def _as_query_component(self, **kwargs: Any) -> "QueryComponent":
        """Get query component."""
        return QueryComponent2()


def test_query_pipeline_chain() -> None:
    """Test query pipeline."""
    # test qc1 by itself with chain syntax
    p = QueryPipeline(chain=[QueryComponent1()])
    output = p.run(input1=1, input2=2)
    # since there's one output, output is just the value
    assert output == 3


def test_query_pipeline_single_arg_inp() -> None:
    """Test query pipeline with single arg input (no kwargs)."""
    # should work if input is a single arg
    p = QueryPipeline(chain=[QueryComponent3(), QueryComponent3()])
    # since there's one output, output is just the value
    output = p.run(3)
    assert output == 12


def test_query_pipeline_input_component() -> None:
    """Test query pipeline input component."""
    # test connecting different inputs to different components
    qc1 = QueryComponent1()
    qc2 = QueryComponent2()
    inp = InputComponent()
    p = QueryPipeline()

    p.add_modules({"qc1": qc1, "qc2": qc2, "inp": inp})
    # add inp.inp1 to both qc1.input1 and qc2.input2
    p.add_link("inp", "qc1", src_key="inp1", dest_key="input1")
    p.add_link("inp", "qc2", src_key="inp1", dest_key="input2")
    # add inp.inp2 to qc1.input2
    p.add_link("inp", "qc1", src_key="inp2", dest_key="input2")
    # add qc1 to qc2.input1
    p.add_link("qc1", "qc2", dest_key="input1")

    output = p.run(inp1=1, inp2=2)
    assert output == "3:1"


def test_query_pipeline_partial() -> None:
    """Test query pipeline."""
    # test qc1 with qc2 with one partial, with chain syntax
    qc1 = QueryComponent1()
    qc2 = QueryComponent2()
    qc2.partial(input2="hello")
    p = QueryPipeline(chain=[qc1, qc2])
    output = p.run(input1=1, input2=2)
    assert output == "3:hello"

    # test qc1 with qc2 with one partial with full syntax
    qc1 = QueryComponent1()
    qc2 = QueryComponent2()
    p = QueryPipeline()
    p.add_modules({"qc1": qc1, "qc2": qc2})
    qc2.partial(input2="foo")
    p.add_link("qc1", "qc2", dest_key="input1")
    output = p.run(input1=2, input2=2)
    assert output == "4:foo"

    # test partial with ChainableMixin
    c2_0 = Chainable2().as_query_component(partial={"input2": "hello"})
    c2_1 = Chainable2().as_query_component(partial={"input2": "world"})
    # you can now define a chain because input2 has been defined
    p = QueryPipeline(chain=[c2_0, c2_1])
    output = p.run(input1=1)
    assert output == "1:hello:world"


def test_query_pipeline_sub() -> None:
    """Test query pipeline."""
    # test qc2 with subpipelines of qc3 w/ full syntax
    qc2 = QueryComponent2()
    qc3 = QueryComponent3()
    p1 = QueryPipeline(chain=[qc3, qc3])
    p = QueryPipeline()
    p.add_modules({"qc2": qc2, "p1": p1})
    # link output of p1 to input1 and input2 of qc2
    p.add_link("p1", "qc2", dest_key="input1")
    p.add_link("p1", "qc2", dest_key="input2")
    output = p.run(input=2)
    assert output == "8:8"


def test_query_pipeline_multi() -> None:
    """Test query pipeline."""
    # try run run_multi
    # link both qc1_0 and qc1_1 to qc2
    qc1_0 = QueryComponent1()
    qc1_1 = QueryComponent1()
    qc2 = QueryComponent2()
    p = QueryPipeline()
    p.add_modules({"qc1_0": qc1_0, "qc1_1": qc1_1, "qc2": qc2})
    p.add_link("qc1_0", "qc2", dest_key="input1")
    p.add_link("qc1_1", "qc2", dest_key="input2")
    output = p.run_multi(
        {"qc1_0": {"input1": 1, "input2": 2}, "qc1_1": {"input1": 3, "input2": 4}}
    )
    assert output == {"qc2": {"output": "3:7"}}


def test_query_pipeline_multi_batch() -> None:
    """Test query pipeline."""
    # try run run_multi
    # link both qc1_0 and qc1_1 to qc2
    qc1_0 = QueryComponent1()
    qc1_1 = QueryComponent1()
    qc2 = QueryComponent2()
    p = QueryPipeline()
    p.add_modules({"qc1_0": qc1_0, "qc1_1": qc1_1, "qc2": qc2})
    p.add_link("qc1_0", "qc2", dest_key="input1")
    p.add_link("qc1_1", "qc2", dest_key="input2")
    output = p.run_multi(
        {
            "qc1_0": {"input1": [1, 5], "input2": [2, 1]},
            "qc1_1": {"input1": [3, 7], "input2": [4, 2]},
        },
        batch=True,
    )
    assert output == {"qc2": {"output": ["3:7", "6:9"]}}


def test_query_pipeline_multi_intermediate_output() -> None:
    """Test query pipeline showing intermediate outputs."""
    # try run run_multi_with_intermediates
    # link both qc1_0 and qc1_1 to qc2
    qc1_0 = QueryComponent1()
    qc1_1 = QueryComponent1()
    qc2 = QueryComponent2()
    p = QueryPipeline()
    p.add_modules({"qc1_0": qc1_0, "qc1_1": qc1_1, "qc2": qc2})
    p.add_link("qc1_0", "qc2", dest_key="input1")
    p.add_link("qc1_1", "qc2", dest_key="input2")
    outputs, intermediates = p.run_multi_with_intermediates(
        {"qc1_0": {"input1": 1, "input2": 2}, "qc1_1": {"input1": 3, "input2": 4}}
    )

    assert outputs == {"qc2": {"output": "3:7"}}

    assert intermediates["qc1_0"].inputs == {"input1": 1, "input2": 2}
    assert intermediates["qc1_0"].outputs == {"output": 3}

    assert intermediates["qc1_1"].inputs == {"input1": 3, "input2": 4}
    assert intermediates["qc1_1"].outputs == {"output": 7}

    assert intermediates["qc2"].inputs == {"input1": 3, "input2": 7}
    assert intermediates["qc2"].outputs == {"output": "3:7"}


@pytest.mark.asyncio
async def test_query_pipeline_async() -> None:
    """Test query pipeline in async fashion."""
    # run some synchronous tests above

    # should work if input is a single arg
    p = QueryPipeline(chain=[QueryComponent3(), QueryComponent3()])
    # since there's one output, output is just the value
    output = await p.arun(3)
    assert output == 12

    # test qc1 with qc2 with one partial with full syntax
    qc1 = QueryComponent1()
    qc2 = QueryComponent2()
    p = QueryPipeline()
    p.add_modules({"qc1": qc1, "qc2": qc2})
    qc2.partial(input2="foo")
    p.add_link("qc1", "qc2", dest_key="input1")
    output = await p.arun(input1=2, input2=2)
    assert output == "4:foo"

    # Test input component
    # test connecting different inputs to different components
    qc1 = QueryComponent1()
    qc2 = QueryComponent2()
    inp = InputComponent()
    p = QueryPipeline()
    p.add_modules({"qc1": qc1, "qc2": qc2, "inp": inp})
    # add inp.inp1 to both qc1.input1 and qc2.input2
    p.add_link("inp", "qc1", src_key="inp1", dest_key="input1")
    p.add_link("inp", "qc2", src_key="inp1", dest_key="input2")
    # add inp.inp2 to qc1.input2
    p.add_link("inp", "qc1", src_key="inp2", dest_key="input2")
    # add qc1 to qc2.input1
    p.add_link("qc1", "qc2", dest_key="input1")
    output = await p.arun(inp1=1, inp2=2)
    assert output == "3:1"

    output = await p.arun(inp1=[1, 2], inp2=[2, 3], batch=True)
    assert output == ["3:1", "5:2"]

    # try run run_multi
    # link both qc1_0 and qc1_1 to qc2
    qc1_0 = QueryComponent1()
    qc1_1 = QueryComponent1()
    qc2 = QueryComponent2()
    p = QueryPipeline()
    p.add_modules({"qc1_0": qc1_0, "qc1_1": qc1_1, "qc2": qc2})
    p.add_link("qc1_0", "qc2", dest_key="input1")
    p.add_link("qc1_1", "qc2", dest_key="input2")
    output = await p.arun_multi(
        {"qc1_0": {"input1": 1, "input2": 2}, "qc1_1": {"input1": 3, "input2": 4}}
    )
    assert output == {"qc2": {"output": "3:7"}}

    output = await p.arun_multi(
        {
            "qc1_0": {"input1": [1, 5], "input2": [2, 1]},
            "qc1_1": {"input1": [3, 7], "input2": [4, 2]},
        },
        batch=True,
    )
    assert output == {"qc2": {"output": ["3:7", "6:9"]}}


def test_query_pipeline_init() -> None:
    """Test query pipeline init params."""
    qc1 = QueryComponent1()
    qc2 = QueryComponent2()
    inp = InputComponent()
    p = QueryPipeline(
        modules={
            "qc1": qc1,
            "qc2": qc2,
            "inp": inp,
        },
        links=[
            Link("inp", "qc1", src_key="inp1", dest_key="input1"),
            Link("inp", "qc2", src_key="inp1", dest_key="input2"),
            Link("inp", "qc1", src_key="inp2", dest_key="input2"),
            Link("qc1", "qc2", dest_key="input1"),
        ],
    )

    output = p.run(inp1=1, inp2=2)
    assert output == "3:1"

    p = QueryPipeline()
    p.add_modules(
        {
            "input": InputComponent(),
            "qc1": QueryComponent1(),
            "qc2": QueryComponent1(),
            "qc3": QueryComponent1(),
        }
    )
    # add links from input
    p.add_links(
        [
            Link("input", "qc1", src_key="inp1", dest_key="input1"),
            Link("input", "qc2", src_key="inp1", dest_key="input1"),
            Link("input", "qc3", src_key="inp1", dest_key="input1"),
        ]
    )
    # add link chain from input through qc1, qc2, q3
    p.add_links(
        [
            Link("input", "qc1", src_key="inp2", dest_key="input2"),
            Link("qc1", "qc2", dest_key="input2"),
            Link("qc2", "qc3", dest_key="input2"),
        ]
    )
    output = p.run(inp2=1, inp1=2)
    assert output == 7


def test_query_pipeline_chain_str() -> None:
    """Test add_chain with only module strings."""
    p = QueryPipeline(
        modules={
            "input": InputComponent(),
            "a": QueryComponent3(),
            "b": QueryComponent3(),
            "c": QueryComponent3(),
            "d": QueryComponent1(),
        }
    )
    p.add_links(
        [
            Link("input", "a", src_key="inp1", dest_key="input"),
            Link("input", "d", src_key="inp2", dest_key="input2"),
            Link("c", "d", dest_key="input1"),
        ]
    )
    p.add_chain(["a", "b", "c"])
    output = p.run(inp1=1, inp2=3)
    assert output == 11


def test_query_pipeline_batch_chain_str() -> None:
    """Test add_chain with only module strings."""
    p = QueryPipeline(
        modules={
            "input": InputComponent(),
            "a": QueryComponent3(),
            "b": QueryComponent3(),
            "c": QueryComponent3(),
            "d": QueryComponent1(),
        }
    )
    p.add_links(
        [
            Link("input", "a", src_key="inp1", dest_key="input"),
            Link("input", "d", src_key="inp2", dest_key="input2"),
            Link("c", "d", dest_key="input1"),
        ]
    )
    p.add_chain(["a", "b", "c"])
    output = p.run(inp1=[1, 5], inp2=[3, 4], batch=True)
    assert output == [11, 44]


def test_query_pipeline_chain_str_intermediate_output() -> None:
    """Test add_chain with only module strings, showing intermediate outputs."""
    p = QueryPipeline(
        modules={
            "input": InputComponent(),
            "a": QueryComponent3(),
            "b": QueryComponent3(),
            "c": QueryComponent3(),
            "d": QueryComponent1(),
        },
    )
    p.add_links(
        [
            Link("input", "a", src_key="inp1", dest_key="input"),
            Link("input", "d", src_key="inp2", dest_key="input2"),
            Link("c", "d", dest_key="input1"),
        ]
    )
    p.add_chain(["a", "b", "c"])
    outputs, intermediates = p.run_with_intermediates(inp1=1, inp2=3)
    assert outputs == 11

    assert intermediates["input"].inputs == {"inp1": 1, "inp2": 3}
    assert intermediates["input"].outputs == {"inp1": 1, "inp2": 3}

    assert intermediates["a"].inputs == {"input": 1}
    assert intermediates["a"].outputs == {"output": 2}

    assert intermediates["b"].inputs == {"input": 2}
    assert intermediates["b"].outputs == {"output": 4}

    assert intermediates["c"].inputs == {"input": 4}
    assert intermediates["c"].outputs == {"output": 8}

    assert intermediates["d"].inputs == {"input1": 8, "input2": 3}
    assert intermediates["d"].outputs == {"output": 11}


def test_query_pipeline_conditional_edges() -> None:
    """Test conditional edges."""

    def choose_fn(input: int) -> Dict:
        """Choose."""
        if input == 1:
            toggle = "true"
        else:
            toggle = "false"
        return {"toggle": toggle, "input": input}

    p = QueryPipeline(
        modules={
            "input": InputComponent(),
            "fn": FnComponent(fn=choose_fn),
            "a": QueryComponent1(),
            "b": QueryComponent2(),
        },
    )

    p.add_links(
        [
            Link("input", "fn", src_key="inp1", dest_key="input"),
            Link("input", "a", src_key="inp2", dest_key="input1"),
            Link("input", "b", src_key="inp2", dest_key="input1"),
            Link(
                "fn",
                "a",
                dest_key="input2",
                condition_fn=lambda x: x["toggle"] == "true",
                input_fn=lambda x: x["input"],
            ),
            Link(
                "fn",
                "b",
                dest_key="input2",
                condition_fn=lambda x: x["toggle"] == "false",
                input_fn=lambda x: x["input"],
            ),
        ]
    )
    output = p.run(inp1=1, inp2=3)
    # should go to a
    assert output == 4

    output = p.run(inp1=2, inp2=3)
    # should go to b
    assert output == "3:2"


def test_query_pipeline_super_conditional() -> None:
    """This tests that paths are properly pruned and maintained for many conditional edges."""

    def simple_fn(val: int):
        print("Running simple_fn", flush=True)
        return val

    def over_twenty_fn(val: int):
        print("Running over_twenty_fn", flush=True)
        return val + 100

    def final_fn(x: int, y: int, z: int):
        print("Running final_fn", flush=True)
        return {
            "x": x,
            "y": y,
            "z": z,
        }

    simple_function_component = FnComponent(fn=simple_fn, output_key="output")
    over_twenty_function_2 = FnComponent(fn=over_twenty_fn, output_key="output")
    final_fn = FnComponent(fn=final_fn, output_key="output")

    qp = QueryPipeline(
        modules={
            "first_decision": simple_function_component,
            "second_decision": simple_function_component,
            "under_ten": simple_function_component,
            "over_twenty": simple_function_component,
            "over_twenty_2": over_twenty_function_2,
            "final": final_fn,
        },
        verbose=True,
    )

    qp.add_link(
        "first_decision",
        "under_ten",
        condition_fn=lambda x: x < 10,
    )
    qp.add_link("under_ten", "final", dest_key="x")
    qp.add_link("under_ten", "final", dest_key="y")
    qp.add_link("under_ten", "final", dest_key="z")

    qp.add_link(
        "first_decision",
        "second_decision",
        condition_fn=lambda x: x >= 10,
    )

    qp.add_link(
        "second_decision",
        "over_twenty",
        condition_fn=lambda x: x > 20,
    )
    qp.add_link(
        "second_decision",
        "over_twenty_2",
        condition_fn=lambda x: x > 20,
    )

    qp.add_link(
        "second_decision",
        "final",
        dest_key="z",
        condition_fn=lambda x: x > 20,
    )
    qp.add_link(
        "over_twenty",
        "final",
        dest_key="x",
    )
    qp.add_link(
        "over_twenty_2",
        "final",
        dest_key="y",
    )

    response = qp.run(val=9)
    assert response == {"x": 9, "y": 9, "z": 9}

    response = qp.run(val=11)
    assert response == 11

    response = qp.run(val=21)
    assert response == {"x": 21, "y": 121, "z": 21}
