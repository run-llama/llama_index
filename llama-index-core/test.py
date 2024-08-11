from typing import Any

from llama_index.core.agent.runner.planner import Plan, SubTask
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms import LLMMetadata, CompletionResponse, CompletionResponseGen


class MockLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata.

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
            ).json()
            return CompletionResponse(text=text)

        # dummy response for react
        return CompletionResponse(text="Final Answer: All done")

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError


if __name__ == "__main__":
    llm = MockLLM()
    print(llm)
