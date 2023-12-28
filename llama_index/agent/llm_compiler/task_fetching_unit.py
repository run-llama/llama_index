"""Task fetching unit.

Taken from 
https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/task_fetching_unit.py

"""

from typing import Dict, Any, List, Optional, Set
from typing import Callable, Collection, Any, Optional
from pydantic import BaseModel, Field
import asyncio
from llama_index.agent.llm_compiler.schema import LLMCompilerTask

SCHEDULING_INTERVAL = 0.01  # seconds


def _replace_arg_mask_with_real_value(
    args, dependencies: List[int], tasks: Dict[str, LLMCompilerTask]
):
    if isinstance(args, (list, tuple)):
        return type(args)(
            _replace_arg_mask_with_real_value(item, dependencies, tasks)
            for item in args
        )
    elif isinstance(args, str):
        for dependency in sorted(dependencies, reverse=True):
            # consider both ${1} and $1 (in case planner makes a mistake)
            for arg_mask in ["${" + str(dependency) + "}", "$" + str(dependency)]:
                if arg_mask in args:
                    if tasks[dependency].observation is not None:
                        args = args.replace(
                            arg_mask, str(tasks[dependency].observation)
                        )
        return args
    else:
        return args

class TaskFetchingUnit(BaseModel):
    """Task fetching unit.

    Detailed in LLMCompiler Paper.
    Code taken from https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/task_fetching_unit.py.
    
    """
    tasks: Dict[str, Task]
    tasks_done: Dict[str, asyncio.Event]
    remaining_tasks: Set[str]

    @classmethod
    def from_tasks(
        cls,
        tasks: Dict[str, Task],
    ):
        """Create a TaskFetchingUnit from a list of tasks."""
        tasks_done = {task_idx: asyncio.Event() for task_idx in tasks}
        remaining_tasks = set(tasks.keys())
        return cls(
            tasks=tasks,
            tasks_done=tasks_done,
            remaining_tasks=remaining_tasks,
    )

    def set_tasks(self, tasks: dict[str, Any]):
        self.tasks.update(tasks)
        self.tasks_done.update({task_idx: asyncio.Event() for task_idx in tasks})
        self.remaining_tasks.update(set(tasks.keys()))

    def _all_tasks_done(self):
        return all(self.tasks_done[d].is_set() for d in self.tasks_done)

    def _get_all_executable_tasks(self):
        return [
            task_name
            for task_name in self.remaining_tasks
            if all(
                self.tasks_done[d].is_set() for d in self.tasks[task_name].dependencies
            )
        ]

    def _preprocess_args(self, task: Task):
        """Replace dependency placeholders, i.e. ${1}, in task.args with the actual observation."""
        args = []
        for arg in task.args:
            arg = _replace_arg_mask_with_real_value(arg, task.dependencies, self.tasks)
            args.append(arg)
        task.args = args

    async def _run_task(self, task: Task):
        self._preprocess_args(task)
        if not task.is_join:
            observation = await task()
            task.observation = observation
        self.tasks_done[task.idx].set()

    async def schedule(self):
        """Run all tasks in self.tasks in parallel, respecting dependencies."""
        # run until all tasks are done
        while not self._all_tasks_done():
            # Find tasks with no dependencies or with all dependencies met
            executable_tasks = self._get_all_executable_tasks()

            for task_name in executable_tasks:
                asyncio.create_task(self._run_task(self.tasks[task_name]))
                self.remaining_tasks.remove(task_name)

            await asyncio.sleep(SCHEDULING_INTERVAL)

    async def aschedule(self, task_queue: asyncio.Queue[Optional[Task]], func):
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
                for task_name in executable_tasks:
                    asyncio.create_task(self._run_task(self.tasks[task_name]))
                    self.remaining_tasks.remove(task_name)
            elif no_more_tasks and self._all_tasks_done():
                # Exit the loop if no more tasks are expected and all tasks are done
                break
            else:
                # If no executable tasks are found, sleep for the SCHEDULING_INTERVAL
                await asyncio.sleep(SCHEDULING_INTERVAL)