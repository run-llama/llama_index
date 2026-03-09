"""
Composition Root style dependency injection for workflows.

Allows workflows to receive a single typed `deps` container at construction
(wiring at the call site) and have steps receive it via `WorkflowContext[DepsT]`,
decoupling step signatures from resource construction.

See: https://github.com/run-llama/llama_index/issues/20900
"""

from __future__ import annotations

import contextvars
from enum import Enum
from typing import Annotated, Any, Generic, TypeVar

from llama_index.core.workflow.resource import Resource
from llama_index.core.workflow.workflow import Workflow

# Context var holding the current DepsWorkflow instance during run/arun.
_current_deps_workflow: contextvars.ContextVar["DepsWorkflow[object] | None"] = (
    contextvars.ContextVar("_current_deps_workflow", default=None)
)


def _get_workflow_deps(**kwargs: object) -> object:
    """Resource factory that returns the current workflow's deps from context."""
    wf = _current_deps_workflow.get()
    if wf is None:
        raise RuntimeError(
            "deps_resource() can only be used with DepsWorkflow when running via .run() or .arun()."
        )
    return wf._composition_root_deps


def deps_resource() -> Resource:
    """
    Resource descriptor for Composition Root deps injection.

    Use in step signatures with WorkflowContext:

        class MyDeps:
            memory: Memory
            llm: LLM

        class MyWorkflow(DepsWorkflow[MyDeps]):
            @step
            async def first(self, ev: StartEvent, deps: WorkflowContext[MyDeps]) -> StopEvent:
                await deps.llm.achat(...)
                return StopEvent(result=...)

    Wire at the call site:

        workflow = MyWorkflow(deps=MyDeps(memory=..., llm=...))
    """
    return Resource(factory=_get_workflow_deps, cache=True)


class DepsScope(str, Enum):
    """
    Lifecycle scope for the dependency container.

    - RUN: One deps instance per run (default).
    - STEP: New instance per step (not typical for composition root).
    - SINGLETON: Single instance shared across workflow instances (use with care).
    """

    RUN = "run"
    STEP = "step"
    SINGLETON = "singleton"


DepsT = TypeVar("DepsT")

# Type alias for step parameters: use as `deps: WorkflowContext[MyDeps]`
WorkflowContext = Annotated[DepsT, deps_resource()]


class DepsWorkflow(Workflow, Generic[DepsT]):
    """
    Workflow base class for Composition Root style DI.

    Accepts a typed `deps` container at construction; steps receive it via
    `deps: WorkflowContext[MyDeps]`. Wiring is done at the call site.

    Example:
        @dataclass
        class MyDeps:
            memory: Memory
            llm: LLM
            customer_id: int

        class MyWorkflow(DepsWorkflow[MyDeps]):
            @step
            async def first(
                self, ev: StartEvent, deps: WorkflowContext[MyDeps]
            ) -> StopEvent:
                response = await deps.llm.achat(...)
                deps.memory.put(response)
                return StopEvent(result=response)

        # Composition root
        workflow = MyWorkflow(deps=MyDeps(
            memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
            llm=OpenAI(model="gpt-4o"),
            customer_id=123,
        ))
        # Testing: pass mocks without changing workflow code
        workflow = MyWorkflow(deps=MyDeps(memory=MockMemory(), llm=MockLLM(), customer_id=999))

    """

    def __init__(
        self,
        *,
        deps: DepsT,
        deps_scope: DepsScope = DepsScope.RUN,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._composition_root_deps = deps
        self._deps_scope = deps_scope

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the workflow with deps available to steps via WorkflowContext."""
        token = _current_deps_workflow.set(self)
        try:
            return super().run(*args, **kwargs)
        finally:
            _current_deps_workflow.reset(token)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Run the workflow asynchronously with deps available to steps."""
        token = _current_deps_workflow.set(self)
        try:
            return await super().arun(*args, **kwargs)
        finally:
            _current_deps_workflow.reset(token)
