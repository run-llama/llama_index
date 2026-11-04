"""Tests for the AnswerRelevancyEvaluator prompt management API."""

from llama_index.core.evaluation.answer_relevancy import AnswerRelevancyEvaluator
from llama_index.core.llms import MockLLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate


def test_get_prompts() -> None:
    evaluator = AnswerRelevancyEvaluator(llm=MockLLM())

    prompts = evaluator.get_prompts()

    assert set(prompts.keys()) == {"eval_template", "refine_template"}
    for prompt in prompts.values():
        assert isinstance(prompt, BasePromptTemplate)


def test_init_refine_template() -> None:
    evaluator = AnswerRelevancyEvaluator(
        llm=MockLLM(),
        refine_template="test refine template: {query_str}",
    )

    prompts = evaluator.get_prompts()
    assert prompts["refine_template"].get_template() == (
        "test refine template: {query_str}"
    )


def test_update_prompts() -> None:
    evaluator = AnswerRelevancyEvaluator(llm=MockLLM())
    new_refine_template = PromptTemplate("new refine template: {query_str}")

    evaluator.update_prompts({"refine_template": new_refine_template})

    prompts = evaluator.get_prompts()
    assert prompts["refine_template"] == new_refine_template
