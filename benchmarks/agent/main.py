from typing import List

import pandas as pd
from agent_utils import AGENTS, ALL_MODELS, get_model, is_valid_combination
from button_tasks import TASKS as BUTTON_TASKS
from fire import Fire
from math_tasks import TASKS as MATH_TASKS

ALL_TASKS = list(MATH_TASKS.keys()) + list(BUTTON_TASKS.keys())


def evaluate(agent: str, model: str, task_name: str, verbose: bool = False) -> bool:
    if task_name in MATH_TASKS:
        task = MATH_TASKS[task_name]()
    elif task_name in BUTTON_TASKS:
        task = BUTTON_TASKS[task_name]()
    else:
        raise ValueError(f"Unknown task {task_name}")

    print("=========================================")
    print(f"Evaluating | {agent} | {model} | {task.message} |")

    llm = get_model(model)
    agent_cls = AGENTS[agent]
    if agent == "react":
        additional_kwargs = {"max_iterations": 10}
    elif agent == "openai":
        additional_kwargs = {"max_function_calls": 10}
    else:
        raise ValueError(f"Unknown agent {agent}")

    agent = agent_cls.from_tools(
        tools=task.tools,
        llm=llm,
        verbose=verbose,
        **additional_kwargs,
    )
    try:
        actual_response = agent.chat(task.message).response
        outcome = task.eval_fn(actual_response, task.expected_response)
    except Exception as e:
        if verbose:
            print(e)

        actual_response = None
        outcome = False

    if verbose:
        print(f"Expected response: {task.expected_response}")
        print(f"Actual response: {actual_response}")
    print(f"Outcome: {outcome}")
    return outcome


def benchmark(
    agents: List[str] = list(AGENTS.keys()),
    models: List[str] = ALL_MODELS,
    tasks: List[str] = ALL_TASKS,
    verbose: bool = False,
    output: str = "results.csv",
    save: bool = True,
) -> pd.DataFrame:
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
    if save:
        df.to_csv(output)
    return df


if __name__ == "__main__":
    Fire(benchmark)
