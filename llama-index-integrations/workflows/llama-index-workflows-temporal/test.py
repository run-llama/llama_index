from temporalio import workflow
with workflow.unsafe.imports_passed_through():
    import asyncio
    from temporalio.client import Client

    from llama_index.core.workflow.events import StartEvent, StopEvent
    from llama_index.core.workflow.workflow import Workflow, step, Event
    from llama_index.workflows.temporal.TemporalWorkflow import LlamaIndexTemporalWorkflow, TemporalWorkflowBuilder, build_wf_activities
    from temporalio.contrib.pydantic import pydantic_data_converter



class EventOne(Event):
    one: str
    pass

class EventTwo(Event):
    two: str
    pass

class TestWorkflow2(Workflow):

    @step()
    async def run(self, start_event: StartEvent) -> EventOne:
        print("run one", start_event)
        return EventOne(one="one")

    @step()
    async def two(self, event_one: EventOne) -> EventTwo:
        print("run two")
        return EventTwo(two="two")

    @step()
    async def three(self, event_two: EventTwo) -> StopEvent:
        print("run three")
        return StopEvent()





async def main():
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    builder = TemporalWorkflowBuilder(client, TestWorkflowContainer, instance)
    handle = await builder.run()
    print(f"handle ran: {handle}")
    pass


instance = TestWorkflow2()
activities = build_wf_activities(TestWorkflow2())

@workflow.defn()
class TestWorkflowContainer(LlamaIndexTemporalWorkflow):

    def __init__(self):
        print("initializing workflow container")
        super().__init__(activities)

    @workflow.run
    async def run(self, start_event: StartEvent):
        print("running workflow container")
        return await super().run(start_event)


if __name__ == "__main__":
    asyncio.run(main())
