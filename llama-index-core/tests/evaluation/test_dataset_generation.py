"""Test dataset generation."""

from llama_index.core.evaluation.dataset_generation import DatasetGenerator
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.schema import TextNode


def test_dataset_generation(patch_llm_predictor) -> None:
    """Test dataset generation."""
    test_nodes = [TextNode(text="hello_world"), TextNode(text="foo_bar")]

    question_gen_prompt = PromptTemplate(
        """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
generate only questions based on the below query.
{query_str}
""",
        prompt_type=PromptType.QUESTION_ANSWER,
    )

    dataset_generator = DatasetGenerator(
        test_nodes,
        text_question_template=question_gen_prompt,
        question_gen_query="gen_question",
    )
    eval_dataset = dataset_generator.generate_dataset_from_nodes()
    qr_pairs = eval_dataset.qr_pairs
    assert len(qr_pairs) == 2
    # the mock LLM concatenates query with context with ":"
    # the first call is to generate the question
    assert qr_pairs[0][0] == "gen_question:hello_world"
    # the second call is to generate the answer
    assert qr_pairs[0][1] == "gen_question:hello_world:hello_world"
    assert qr_pairs[1][0] == "gen_question:foo_bar"
    assert qr_pairs[1][1] == "gen_question:foo_bar:foo_bar"
