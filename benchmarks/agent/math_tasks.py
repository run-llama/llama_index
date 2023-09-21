from typing import Callable, Dict

from eval import contains_expected_response
from task import Task

from llama_index.tools.function_tool import FunctionTool


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)


POWER_TASK = Task(
    message="What is 3 to the power of 4?",
    expected_response="81",
    tools=[add_tool, multiply_tool],
    eval_fn=contains_expected_response,
)

MULTIPLY_THEN_ADD_TASK = Task(
    message="What is 123 + 321 * 2?",
    expected_response="765",
    tools=[add_tool, multiply_tool],
    eval_fn=contains_expected_response,
)


TASKS: Dict[str, Callable[..., Task]] = {
    "multiply_then_add": lambda: MULTIPLY_THEN_ADD_TASK,
    "power": lambda: POWER_TASK,
}
