from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation.faithfulness import DEFAULT_EVAL_TEMPLATE
from llama_index.core.llms import MockLLM
from llama_index.core.prompts import PromptTemplate

CUSTOM_EVAL_TEMPLATE = (
    "CUSTOM MARKER: is {query_str} supported by {context_str}? Answer YES or NO."
)


def test_faithfulness_eval_template_from_str() -> None:
    evaluator = FaithfulnessEvaluator(llm=MockLLM(), eval_template=CUSTOM_EVAL_TEMPLATE)
    assert (
        evaluator.get_prompts()["eval_template"].get_template() == CUSTOM_EVAL_TEMPLATE
    )


def test_faithfulness_eval_template_from_base_prompt_template() -> None:
    evaluator = FaithfulnessEvaluator(
        llm=MockLLM(), eval_template=PromptTemplate(CUSTOM_EVAL_TEMPLATE)
    )
    assert (
        evaluator.get_prompts()["eval_template"].get_template() == CUSTOM_EVAL_TEMPLATE
    )


def test_faithfulness_eval_template_default_when_not_provided() -> None:
    evaluator = FaithfulnessEvaluator(llm=MockLLM())
    assert (
        evaluator.get_prompts()["eval_template"].get_template()
        == DEFAULT_EVAL_TEMPLATE.get_template()
    )
