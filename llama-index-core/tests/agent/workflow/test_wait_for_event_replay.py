import pytest

from llama_index.core.workflow import (
    Context,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


@pytest.mark.asyncio
async def test_wait_for_event_replay_reuses_persisted_side_effect() -> None:
    class ReplaySafeWorkflow(Workflow):
        @step
        async def wait_for_approval(self, ctx: Context, ev: StartEvent) -> StopEvent:
            step_entries = await ctx.store.get("step_entries", default=0)
            await ctx.store.set("step_entries", step_entries + 1)

            cache_key = "approval:request-1"
            side_effect_result = await ctx.store.get(cache_key, default=None)
            if side_effect_result is None:
                calls = await ctx.store.get("side_effect_calls", default=0)
                await ctx.store.set("side_effect_calls", calls + 1)
                side_effect_result = "request inspected"
                await ctx.store.set(cache_key, side_effect_result)

            waiter_id = "approval:request-1"
            response = await ctx.wait_for_event(
                HumanResponseEvent,
                waiter_id=waiter_id,
                waiter_event=InputRequiredEvent(
                    prefix="Approve request?",
                    waiter_id=waiter_id,
                ),
                requirements={"waiter_id": waiter_id},
            )
            return StopEvent(result=(side_effect_result, response.response))

    handler = ReplaySafeWorkflow().run()
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            handler.ctx.send_event(
                HumanResponseEvent(response="yes", waiter_id=event.waiter_id)
            )

    assert await handler == ("request inspected", "yes")
    assert await handler.ctx.store.get("step_entries") == 2
    assert await handler.ctx.store.get("side_effect_calls") == 1
