from typing import Callable, Dict, List

from task import Task

from llama_index.tools.function_tool import FunctionTool


class Phone:
    def __init__(self) -> None:
        self.number = ""
        self.entered = False

    def dial_digit(self, number: str):
        """Dial a digit  on the phone."""
        assert len(number) == 1 and number.isdigit()
        self.number += number

    def enter(self):
        """Press the enter key on the phone."""
        if self.entered:
            raise Exception("Already entered")

        self.entered = True

    def evaluate(self, response, expected_response: str) -> bool:
        return self.number == expected_response and self.entered


def search_number(first_name: str, last_name: str) -> str:
    """Search for a person by first and last name."""
    if first_name == "John" and last_name == "Smith":
        return "2135"
    else:
        return "No results found. Please capitalize both first and last name."


search_number_tool = FunctionTool.from_defaults(fn=search_number)


def get_dial_then_enter() -> Task:
    phone = Phone()

    dial_digit_tool = FunctionTool.from_defaults(fn=phone.dial_digit)
    enter_tool = FunctionTool.from_defaults(fn=phone.enter)

    task = Task(
        message="Dial the number 4151 then hit enter.",
        expected_response="4151",
        tools=[dial_digit_tool, enter_tool],
        eval_fn=phone.evaluate,
    )
    return task


def get_search_then_dial() -> Task:
    phone = Phone()

    dial_digit_tool = FunctionTool.from_defaults(fn=phone.dial_digit)
    enter_tool = FunctionTool.from_defaults(fn=phone.enter)

    task = Task(
        message="Dial the number for john smith, then hit enter.",
        expected_response="2135",
        tools=[dial_digit_tool, enter_tool, search_number_tool],
        eval_fn=phone.evaluate,
    )
    return task


TASKS: Dict[str, Callable[..., Task]] = {
    "dial_then_enter": get_dial_then_enter,
    "search_then_dial": get_search_then_dial,
}
