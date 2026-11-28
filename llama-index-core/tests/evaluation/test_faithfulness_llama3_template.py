from typing import Any, List

from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


class RecordingLLM(CustomLLM):
    """Fake LLM that reports itself as llama3:8b and records every prompt."""

    prompts: List[str] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="llama3:8b")

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.prompts.append(prompt)
        return CompletionResponse(text="YES")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        raise NotImplementedError


def test_llama3_template_includes_response_and_contexts() -> None:
    llm = RecordingLLM(prompts=[])
    evaluator = FaithfulnessEvaluator(llm=llm)
    response = "The Eiffel Tower is in Paris."
    context = "The Eiffel Tower stands in the city of Paris, France."

    evaluator.evaluate(response=response, contexts=[context])

    prompt = llm.prompts[-1]
    assert response in prompt
    assert context in prompt
    assert "{query}" not in prompt
    assert "{reference_contexts}" not in prompt
