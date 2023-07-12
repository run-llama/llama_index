from typing import List

import pandas as pd
from fire import Fire
from agent_utils import (
    AGENTS,
    ALL_MODELS,
    get_model,
    is_valid_combination,
)
import button_tasks
import math_tasks
from math_tasks import TASKS as MATH_TASKS
from button_tasks import TASKS as BUTTON_TASKS
from task import Task

ALL_TASKS = MATH_TASKS + BUTTON_TASKS


def evaluate(agent: str, model: str, task_name: str, verbose: bool = False) -> bool:
    if task_name in MATH_TASKS:
        task = math_tasks.get_tasks([task_name])[0]
    elif task_name in BUTTON_TASKS:
        task = button_tasks.get_tasks([task_name])[0]
    print("=====")
    print(f"| Evaluating | {agent} | {model} | {task.message} |")

    llm = get_model(model)
    agent_cls = AGENTS[agent]
    agent = agent_cls.from_tools(
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
    agents: List[str] = list(AGENTS.keys()),
    models: List[str] = ALL_MODELS,
    tasks: List[Task] = ALL_TASKS,
    verbose: bool = False,
) -> None:
    data = []
    for agent in agents:
        for model in models:
            for task in tasks:
                if not is_valid_combination(agent, model):
                    continue
                outcome = evaluate(agent, model, task, verbose)
                data.append(
                    {
                        "agent": agent,
                        "model": model,
                        "task": task,
                        "outcome": outcome,
                    }
                )
    df = pd.DataFrame(data)
    df.to_csv("results.csv")


if __name__ == "__main__":
    Fire(main)
