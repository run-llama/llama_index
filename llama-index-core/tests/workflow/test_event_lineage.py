import asyncio

import pytest
from workflows import Context, Workflow, step
from workflows.events import (
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)


class StaleReplayWorkflow(Workflow):
    """A workflow that waits for a human approval event.

    Without requirements, any HumanResponseEvent can satisfy the waiter.
    With requirements, the waiter validates proposal_id, reviewer, and approved_action.
    """

    def __init__(self, guarded: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.guarded = guarded

    @step
    async def review_and_execute(self, ctx: Context, ev: StartEvent) -> StopEvent:
        proposal_id = ev.proposal_id
        requested_action = ev.requested_action
        reviewer = ev.reviewer

        requirements = None
        if self.guarded:
            requirements = {
                "proposal_id": proposal_id,
                "reviewer": reviewer,
                "approved_action": requested_action,
            }

        response = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_event=InputRequiredEvent(
                prompt="Approve?",
                proposal_id=proposal_id,
                reviewer=reviewer,
                requested_action=requested_action,
            ),
            waiter_id=proposal_id,
            requirements=requirements,
            timeout=5,
        )

        return StopEvent(
            result={
                "consumed_proposal_id": response.get("proposal_id"),
                "consumed_approved_action": response.get("approved_action"),
                "response": response.response,
                "expected_proposal_id": proposal_id,
            }
        )


async def _run_with_events(
    workflow: Workflow,
    start_kwargs: dict,
    events_to_send: list,
) -> dict:
    handler = workflow.run(**start_kwargs)

    stream_task = asyncio.create_task(_drain_stream(handler))
    await asyncio.sleep(0.1)

    for event in events_to_send:
        await handler.send_event(event)
        await asyncio.sleep(0.1)

    result = await handler
    await stream_task
    return result


async def _drain_stream(handler):
    async for _ in handler.stream_events():
        pass


class TestEventLineageSafety:
    """Demonstrate safe event consumption patterns with wait_for_event requirements."""

    @pytest.mark.asyncio
    async def test_requirements_block_stale_replay(self):
        """A guarded waiter bound to proposal lineage rejects a stale event."""
        current = {
            "proposal_id": "req-001",
            "requested_action": "delete_record",
            "reviewer": "alice",
        }
        stale_event = HumanResponseEvent(
            response="approve",
            proposal_id="req-000",  # different from current
            reviewer="alice",
            approved_action="search_db",
            approved_args={"query": "users:42"},
        )
        valid_event = HumanResponseEvent(
            response="approve",
            proposal_id="req-001",
            reviewer="alice",
            approved_action="delete_record",
            approved_args={"table": "users", "record_id": 42},
        )

        workflow = StaleReplayWorkflow(guarded=True, timeout=10)
        result = await _run_with_events(workflow, current, [stale_event, valid_event])

        assert result["consumed_proposal_id"] == "req-001"
        assert result["consumed_approved_action"] == "delete_record"

    @pytest.mark.asyncio
    async def test_no_requirements_allows_stale_event(self):
        """Without requirements, a stale event satisfies the waiter by type alone."""
        current = {
            "proposal_id": "req-001",
            "requested_action": "delete_record",
            "reviewer": "alice",
        }
        stale_event = HumanResponseEvent(
            response="approve",
            proposal_id="req-000",
            reviewer="alice",
            approved_action="search_db",
            approved_args={"query": "users:42"},
        )

        workflow = StaleReplayWorkflow(guarded=False, timeout=10)
        result = await _run_with_events(workflow, current, [stale_event])

        # Without requirements, the stale event is consumed as if it were current
        assert result["consumed_proposal_id"] == "req-000"
        assert result["response"] == "approve"
