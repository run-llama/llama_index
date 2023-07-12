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


def get_dial_then_enter() -> Task:
    phone = Phone()

    dial_digit_tool = FunctionTool.from_defaults(fn=phone.dial_digit)
    enter_tool = FunctionTool.from_defaults(fn=phone.enter)

    task = Task(
        message="Dial the number 4151 then hit enter.",
        expected_response="4151",
        tools=[dial_digit_tool, enter_tool],
        eval_fn=lambda response, expected: phone.number == expected and phone.entered,
    )
    return task


TASKS: Dict[str, Callable[..., Task]] = {"dial_then_enter": get_dial_then_enter}
