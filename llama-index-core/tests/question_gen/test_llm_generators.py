from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata


def test_llm_question_gen(patch_llm_predictor, patch_token_text_splitter) -> None:
    question_gen = LLMQuestionGenerator.from_defaults()
    tools = [
        ToolMetadata(description="data source 1", name="source_1"),
        ToolMetadata(description="data source 2", name="source_2"),
    ]
    query = QueryBundle(query_str="What is A and B?")
    sub_questions = question_gen.generate(tools=tools, query=query)
    assert isinstance(sub_questions[0], SubQuestion)
