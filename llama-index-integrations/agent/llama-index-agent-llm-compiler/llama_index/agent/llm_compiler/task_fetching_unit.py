"""Task fetching unit.

Taken from
https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/task_fetching_unit.py

"""

import asyncio
from typing import Any, Collection, Dict, List, Set, Tuple, Union

from llama_index.core.utils import print_text
from pydantic import BaseModel

from .schema import LLMCompilerTask
from .utils import parse_llm_compiler_action_args

SCHEDULING_INTERVAL = 0.01  # seconds


def _replace_arg_mask_with_real_value(
    args: Union[List, Tuple, str],
    dependencies: Collection[int],
    tasks: Dict[int, LLMCompilerTask],
) -> Union[List, Tuple]:
    if isinstance(args, (list, tuple)):
        new_list: List[Any] = []
        for item in args:
            new_item = _replace_arg_mask_with_real_value(item, dependencies, tasks)
            # if the original item was string but the new item is not, then treat it as expanded
            # arguments.
            # hack to get around ast.literal_eval not being able to parse strings with template variables
            # e.g. "$1, 2" -> ("$1, 2", )
            if isinstance(item, str) and not isinstance(new_item, str):
                new_list.extend(new_item)
            else:
                new_list.append(new_item)
        return type(args)(new_list)
    elif isinstance(args, str):
        for dependency in sorted(dependencies, reverse=True):
            # consider both ${1} and $1 (in case planner makes a mistake)
            for arg_mask in ["${" + str(dependency) + "}", "$" + str(dependency)]:
                if arg_mask in args:
                    if tasks[dependency].observation is not None:
                        args = args.replace(
                            arg_mask, str(tasks[dependency].observation)
                        )

        # need to re-call parse_llm_compiler_action_args after replacement,
        # this is because arg strings with template variables get formatted
        # into lists (ast.literal_eval fails):
        # e.g. "$1, 2" -> ("$1, 2", )
        # so after replacement need to rerun this
        return parse_llm_compiler_action_args(args)
    else:
        return args


class TaskFetchingUnit(BaseModel):
    """Task fetching unit.

    Detailed in LLMCompiler Paper.
    Code taken from https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/task_fetching_unit.py.

    """

    tasks: Dict[int, LLMCompilerTask]
    tasks_done: Dict[int, asyncio.Event]
    remaining_tasks: Set[int]
    verbose: bool = False

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_tasks(
        cls,
        tasks: Dict[int, LLMCompilerTask],
        verbose: bool = False,
    ) -> "TaskFetchingUnit":
        """Create a TaskFetchingUnit from a list of tasks."""
        tasks_done = {task_idx: asyncio.Event() for task_idx in tasks}
        remaining_tasks = set(tasks.keys())
        return cls(
            tasks=tasks,
            tasks_done=tasks_done,
            remaining_tasks=remaining_tasks,
            verbose=verbose,
        )

    def set_tasks(self, tasks: Dict[int, Any]) -> None:
        self.tasks.update(tasks)
        self.tasks_done.update({task_idx: asyncio.Event() for task_idx in tasks})
        self.remaining_tasks.update(set(tasks.keys()))

    def _all_tasks_done(self) -> bool:
        return all(self.tasks_done[d].is_set() for d in self.tasks_done)

    def _get_all_executable_tasks(self) -> List[int]:
        return [
            task_id
            for task_id in self.remaining_tasks
            if all(
                self.tasks_done[d].is_set() for d in self.tasks[task_id].dependencies
            )
        ]

    def _preprocess_args(self, task: LLMCompilerTask) -> None:
        """Replace dependency placeholders, i.e. ${1}, in task.args with the actual observation."""
        args = _replace_arg_mask_with_real_value(
            task.args, task.dependencies, self.tasks
        )
        task.args = args

    async def _run_task(self, task: LLMCompilerTask) -> None:
        self._preprocess_args(task)
        if not task.is_join:
            observation = await task()
            task.observation = observation
        if self.verbose:
            print_text(
                f"Ran task: {task.name}. Observation: {task.observation}\n",
                color="blue",
            )
        self.tasks_done[task.idx].set()

    async def schedule(self) -> None:
        """Run all tasks in self.tasks in parallel, respecting dependencies."""
        # run until all tasks are done
        while not self._all_tasks_done():
            # Find tasks with no dependencies or with all dependencies met
            executable_tasks = self._get_all_executable_tasks()

            async_tasks = []
            for task_id in executable_tasks:
                async_tasks.append(self._run_task(self.tasks[task_id]))
                self.remaining_tasks.remove(task_id)
            await asyncio.gather(*async_tasks)

            await asyncio.sleep(SCHEDULING_INTERVAL)

    async def aschedule(self, task_queue: asyncio.Queue) -> None:
        """Asynchronously listen to task_queue and schedule tasks as they arrive."""
        no_more_tasks = False  # Flag to check if all tasks are received

        while True:
            if not no_more_tasks:
                # Wait for a new task to be added to the queue
                task = await task_queue.get()

                # Check for sentinel value indicating end of tasks
                if task is None:
                    no_more_tasks = True
                else:
                    # Parse and set the new tasks
                    self.set_tasks({task.idx: task})

            # Schedule and run executable tasks
            executable_tasks = self._get_all_executable_tasks()

            if executable_tasks:
                for task_id in executable_tasks:
                    asyncio.create_task(self._run_task(self.tasks[task_id]))
                    self.remaining_tasks.remove(task_id)
            elif no_more_tasks and self._all_tasks_done():
                # Exit the loop if no more tasks are expected and all tasks are done
                break
            else:
                # If no executable tasks are found, sleep for the SCHEDULING_INTERVAL
                await asyncio.sleep(SCHEDULING_INTERVAL)
