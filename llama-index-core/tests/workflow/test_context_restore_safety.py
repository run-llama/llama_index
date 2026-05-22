"""Tests demonstrating safe Context restore patterns for serialized workflow state.

When Context.to_dict() / Context.from_dict() is used to checkpoint and restore
workflow state, the serialized payload can be altered before restore. Application
code should validate restored context fields against trusted request lineage
rather than blindly trusting the restored values.
"""

from dataclasses import dataclass
from typing import Any

import pytest
from workflows import Context, Workflow, step
from workflows.events import (
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)


@dataclass
class RestoredContextFixture:
    """Simulates a serialized-then-restored context with approval-bearing fields."""

    proposal_id: str
    approved_action: str
    approved_args: dict[str, Any]
    requested_action: str
    requested_args: dict[str, Any]


# A clean restore where approved_action matches the requested action
CLEAN_RESTORE = RestoredContextFixture(
    proposal_id="proposal-001",
    approved_action="search_db",
    approved_args={"query": "users:42"},
    requested_action="search_db",
    requested_args={"query": "users:42"},
)

# A tampered restore where approved_action was changed to a privileged operation
TAMPERED_RESTORE = RestoredContextFixture(
    proposal_id="proposal-001",
    approved_action="delete_record",
    approved_args={"table": "users", "record_id": 42},
    requested_action="search_db",
    requested_args={"query": "users:42"},
)

HIGH_PRIVILEGE_ACTIONS = {"delete_record", "update_config"}


class ContextRestoreWorkflow(Workflow):
    """A workflow that restores context from a serialized fixture.

    The `guarded` flag determines whether the workflow validates restored
    approval-bearing fields against the server-side request ticket before
    executing the action.
    """

    def __init__(self, restore_fixture: RestoredContextFixture, guarded: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._fixture = restore_fixture
        self.guarded = guarded

    @step
    async def approve_and_execute(self, ctx: Context, ev: StartEvent) -> StopEvent:
        # Simulate restoring serialized context
        restored = self._fixture

        if self.guarded:
            # SAFE: validate restored action fields against the request ticket
            if restored.approved_action != restored.requested_action:
                return StopEvent(
                    result={
                        "executed": False,
                        "action": "none",
                        "reason": (
                            f"Restored approved_action={restored.approved_action} "
                            f"does not match requested_action={restored.requested_action}"
                        ),
                    }
                )

        # UNSAFE (or safe if validation passed): execute the restored action
        return StopEvent(
            result={
                "executed": True,
                "action": restored.approved_action,
                "args": restored.approved_args,
                "privilege": (
                    "high"
                    if restored.approved_action in HIGH_PRIVILEGE_ACTIONS
                    else "low"
                ),
            }
        )


class TestContextRestoreSafety:
    @pytest.mark.asyncio
    async def test_guarded_restore_rejects_tampered_action(self):
        """A guarded restore rejects a tampered approved_action."""
        workflow = ContextRestoreWorkflow(TAMPERED_RESTORE, guarded=True, timeout=5)
        handler = workflow.run(proposal_id="proposal-001")

        result = await handler
        assert result["executed"] is False
        assert result["action"] == "none"

    @pytest.mark.asyncio
    async def test_guarded_restore_accepts_matching_action(self):
        """A guarded restore accepts an approved_action that matches the request."""
        workflow = ContextRestoreWorkflow(CLEAN_RESTORE, guarded=True, timeout=5)
        handler = workflow.run(proposal_id="proposal-001")

        result = await handler
        assert result["executed"] is True
        assert result["action"] == "search_db"
        assert result["privilege"] == "low"

    @pytest.mark.asyncio
    async def test_unguarded_restore_executes_tampered_action(self):
        """Without validation, a tampered restore executes the attacker's action."""
        workflow = ContextRestoreWorkflow(TAMPERED_RESTORE, guarded=False, timeout=5)
        handler = workflow.run(proposal_id="proposal-001")

        result = await handler
        assert result["executed"] is True
        assert result["action"] == "delete_record"
        assert result["privilege"] == "high"
