from typing import Any

from llama_index.core.workflow.context import Context


class WorkflowResult:
    def __init__(self, ctx: Context, result: Any, is_done: bool = True) -> None:
        self.ctx = ctx
        self.result = result
        self.is_done = is_done

    def __str__(self) -> str:
        return str(self.result)
