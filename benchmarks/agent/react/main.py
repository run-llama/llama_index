from typing import Callable, List

import pandas as pd
from fire import Fire
from pydantic import BaseModel

from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from llama_index.tools import BaseTool, FunctionTool


class Task(BaseModel):
    message: str
    expected_response: str
    tools: List[BaseTool]
    eval_fn: Callable[[str, str], bool]

    class Config:
        arbitrary_types_allowed = True


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)


def contains_expected_response(response: str, expected_response: str) -> bool:
    """Check if the response contains the expected response."""
    return expected_response in response


MODELS = [
    "text-davinci-003",
    "gpt-3.5-turbo",
    "gpt-4",
]

TASKS = [
    Task(
        message="What is 123 + 321 * 2?",
        expected_response="765",
        tools=[add_tool, multiply_tool],
        eval_fn=contains_expected_response,
    ),
]


def evaluate(model: str, task: Task, verbose: bool = False) -> bool:
    print("=====")
    print(f"Evaluating {model} on {task.message}")

    llm = OpenAI(model=model)
    agent = ReActAgent.from_tools(
        tools=task.tools,
        llm=llm,
        verbose=verbose,
    )
    try:
        actual_response = agent.chat(task.message).response
        outcome = task.eval_fn(actual_response, task.expected_response)
    except Exception as e:
        if verbose:
            print(e)

        actual_response = None
        outcome = False

    print(f"Expected response: {task.expected_response}")
    print(f"Actual response: {actual_response}")
    print(f"Outcome: {outcome}")
    print("=====")
    return outcome


def main(
    models: List[str] = MODELS, tasks: List[Task] = TASKS, verbose: bool = False
) -> None:
    data = []
    for model in models:
        for task in tasks:
            outcome = evaluate(model, task, verbose)
            data.append(
                {
                    "model": model,
                    "task": task.message,
                    "outcome": outcome,
                }
            )
    df = pd.DataFrame(data)
    df.to_csv("results.csv")


if __name__ == "__main__":
    Fire(main)
