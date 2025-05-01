from typing import Any

from llama_index.core.agent import ReActAgentWorker, StructuredPlannerAgent
from llama_index.core.agent.runner.planner import Plan, SubTask
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms import LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.core.tools import FunctionTool


class MockLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        """
        LLM metadata.

        Returns:
            LLMMetadata: LLM metadata containing various information about the LLM.

        """
        return LLMMetadata()

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        if "CREATE A PLAN" in prompt:
            text = Plan(
                sub_tasks=[
                    SubTask(
                        name="one", input="one", expected_output="one", dependencies=[]
                    ),
                    SubTask(
                        name="two", input="two", expected_output="two", dependencies=[]
                    ),
                    SubTask(
                        name="three",
                        input="three",
                        expected_output="three",
                        dependencies=["one", "two"],
                    ),
                ]
            ).model_dump_json()
            return CompletionResponse(text=text)

        # dummy response for react
        return CompletionResponse(text="Final Answer: All done")

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError


def dummy_function(a: int, b: int) -> int:
    """A dummy function that adds two numbers together."""
    return a + b


def test_planner_agent() -> None:
    dummy_tool = FunctionTool.from_defaults(fn=dummy_function)
    dummy_llm = MockLLM()

    worker = ReActAgentWorker.from_tools([dummy_tool], llm=dummy_llm)
    agent = StructuredPlannerAgent(worker, tools=[dummy_tool], llm=dummy_llm)

    # create a plan
    plan_id = agent.create_plan("CREATE A PLAN")
    plan = agent.state.plan_dict[plan_id]
    assert plan is not None
    assert len(plan.sub_tasks) == 3
    assert len(agent.state.get_completed_sub_tasks(plan_id)) == 0
    assert len(agent.state.get_remaining_subtasks(plan_id)) == 3
    assert len(agent.state.get_next_sub_tasks(plan_id)) == 2

    next_tasks = agent.state.get_next_sub_tasks(plan_id)

    for task in next_tasks:
        response = agent.run_task(task.name)
        agent.state.add_completed_sub_task(plan_id, task)

    assert len(agent.state.get_completed_sub_tasks(plan_id)) == 2

    next_tasks = agent.state.get_next_sub_tasks(plan_id)
    assert len(next_tasks) == 1

    # will insert the original dummy plan again
    agent.refine_plan("CREATE A PLAN", plan_id)

    assert len(plan.sub_tasks) == 3
    assert len(agent.state.get_completed_sub_tasks(plan_id)) == 2
    assert len(agent.state.get_remaining_subtasks(plan_id)) == 1
    assert len(agent.state.get_next_sub_tasks(plan_id)) == 1
