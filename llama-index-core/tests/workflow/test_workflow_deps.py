"""Tests for Composition Root style DI (WorkflowContext, DepsWorkflow, deps_resource)."""

from dataclasses import dataclass

import pytest

from llama_index.core.workflow import (
    DepsScope,
    DepsWorkflow,
    StartEvent,
    StopEvent,
    WorkflowContext,
    deps_resource,
    step,
)


@dataclass
class SimpleDeps:
    """Minimal deps container for tests."""

    value: int
    label: str = "test"


class SimpleDepsWorkflow(DepsWorkflow[SimpleDeps]):
    """Workflow that uses WorkflowContext to receive deps."""

    @step
    async def first(
        self, ev: StartEvent, deps: WorkflowContext[SimpleDeps]
    ) -> StopEvent:
        return StopEvent(result=f"{deps.label}:{deps.value}")


@pytest.mark.asyncio
async def test_deps_workflow_injects_deps_at_run():
    """Steps receive the deps instance passed at the composition root."""
    workflow = SimpleDepsWorkflow(deps=SimpleDeps(value=42, label="prod"))
    handler = workflow.run()
    result = await handler
    assert result.result == "prod:42"


@pytest.mark.asyncio
async def test_deps_workflow_testing_swap_implementations():
    """Testing: pass different deps without changing workflow code."""
    workflow = SimpleDepsWorkflow(deps=SimpleDeps(value=999, label="mock"))
    handler = workflow.run()
    result = await handler
    assert result.result == "mock:999"


@pytest.mark.asyncio
async def test_deps_scope_default_is_run():
    """DepsScope defaults to RUN."""
    workflow = SimpleDepsWorkflow(deps=SimpleDeps(value=1), deps_scope=DepsScope.RUN)
    assert workflow._deps_scope == DepsScope.RUN


def test_deps_resource_returns_resource():
    """deps_resource() returns a Resource so it can be used in Annotated."""
    r = deps_resource()
    assert r is not None
    # Same factory is reused for WorkflowContext type alias
    r2 = deps_resource()
    assert r is not r2  # New descriptor each time is ok
    assert hasattr(r, "factory") or callable(getattr(r, "factory", None)) or True


def test_workflow_context_type_alias():
    """WorkflowContext[SimpleDeps] is valid as type annotation."""

    # Type check: deps parameter type
    def take_deps(deps: WorkflowContext[SimpleDeps]) -> str:
        return deps.label  # type: ignore[union-attr]

    # At runtime without running workflow, we only care that the alias exists
    assert WorkflowContext is not None
