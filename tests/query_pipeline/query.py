"""Query pipeline."""

from typing import Any, Dict

from llama_index.core.query_pipeline.query_component import (
    InputKeys,
    OutputKeys,
    QueryComponent,
)
from llama_index.query_pipeline.query import QueryPipeline


class QueryComponent1(QueryComponent):
    """Query component 1.

    Adds two numbers together.

    """

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""
        pass

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

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"input1", "input2"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


class QueryComponent2(QueryComponent):
    """Query component 1.

    Joins two strings together with ':'

    """

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""
        pass

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

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"input1", "input2"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


class QueryComponent3(QueryComponent):
    """Query component 3.

    Takes one input and doubles it.

    """

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""
        pass

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "input" not in input:
            raise ValueError("input not in input")
        return input

    def _run_component(self, **kwargs: Any) -> Dict:
        """Run component."""
        return {"output": kwargs["input"] + kwargs["input"]}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"input"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


def test_query_pipeline_chain():
    """Test query pipeline."""
    # test qc1 by itself with chain syntax
    p = QueryPipeline(chain=[QueryComponent1()])
    output = p.run(input1=1, input2=2)
    # since there's one output, output is just the value
    assert output == 3


def test_query_pipeline_partial():
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


def test_query_pipeline_sub():
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


def test_query_pipeline_multi():
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
