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


def test_dataset_generation_num_limit_and_engine_reuse(
    monkeypatch, patch_llm_predictor
) -> None:
    """Ensure DatasetGenerator reuses the QA engine per node and skips excess QA queries beyond `num`."""
    from llama_index.core.indices.list import SummaryIndex

    test_nodes = [
        TextNode(text="1. q1\n2. q2\n3. q3"),
        TextNode(text="1. q4\n2. q5"),
    ]
    question_gen_prompt = PromptTemplate(
        "{context_str}",
        prompt_type=PromptType.QUESTION_ANSWER,
    )
    qa_prompt = PromptTemplate(
        "{query_str}:{context_str}",
        prompt_type=PromptType.QUESTION_ANSWER,
    )

    orig_as_query_engine = SummaryIndex.as_query_engine
    as_query_engine_calls = 0
    qa_aquery_calls = 0

    def counting_as_query_engine(self, *args, **kwargs):
        nonlocal as_query_engine_calls, qa_aquery_calls
        as_query_engine_calls += 1
        engine = orig_as_query_engine(self, *args, **kwargs)
        if kwargs.get("text_qa_template") is qa_prompt:
            orig_aquery = engine.aquery

            async def counting_aquery(query):
                nonlocal qa_aquery_calls
                qa_aquery_calls += 1
                return await orig_aquery(query)

            monkeypatch.setattr(engine, "aquery", counting_aquery)
        return engine

    monkeypatch.setattr(SummaryIndex, "as_query_engine", counting_as_query_engine)

    dataset_generator = DatasetGenerator(
        test_nodes,
        text_question_template=question_gen_prompt,
        text_qa_template=qa_prompt,
        question_gen_query="gen_question",
    )
    eval_dataset = dataset_generator.generate_dataset_from_nodes(num=2)
    qr_pairs = eval_dataset.qr_pairs

    assert [q for q, _ in qr_pairs] == ["gen_question:1. q1", "q2"]
    # 2 question-gen engines (for the 2 nodes) + 1 QA engine (only node 0 needed for num=2)
    assert as_query_engine_calls == 3
    # Only the 2 retained questions should trigger QA queries (not all 5 generated questions)
    assert qa_aquery_calls == 2
