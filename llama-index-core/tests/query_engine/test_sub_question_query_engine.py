from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks import CallbackManager
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.query_engine.sub_question_query_engine import (
    SubQuestionQueryEngine,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def _make_tool(name: str, raises=None):
    engine = MagicMock()
    engine.callback_manager = CallbackManager([])
    resp = Response(response=f"answer from {name}", source_nodes=[])
    if raises:
        engine.query.side_effect = raises
        engine.aquery = AsyncMock(side_effect=raises)
    else:
        engine.query.return_value = resp
        engine.aquery = AsyncMock(return_value=resp)
    return QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(name=name, description=name),
    )


def _make_components():
    tools = [
        _make_tool("good_engine"),
        _make_tool("bad_engine", raises=RuntimeError("API rate limit exceeded")),
    ]
    sub_questions = [
        SubQuestion(sub_question="q1", tool_name="good_engine"),
        SubQuestion(sub_question="q2", tool_name="bad_engine"),
    ]
    question_gen = MagicMock()
    question_gen.generate.return_value = sub_questions
    question_gen.agenerate = AsyncMock(return_value=sub_questions)

    synth = MagicMock()
    synth.synthesize.return_value = Response(response="final")
    synth.asynthesize = AsyncMock(return_value=Response(response="final"))

    return tools, question_gen, synth


def test_partial_failure_sync():
    """use_async=False: RuntimeError from one engine should not crash the query."""
    tools, question_gen, synth = _make_components()
    engine = SubQuestionQueryEngine(
        question_gen=question_gen,
        response_synthesizer=synth,
        query_engine_tools=tools,
        use_async=False,
        verbose=False,
    )

    response = engine.query(QueryBundle(query_str="compare sources"))

    assert str(response) == "final"
    tools[0].query_engine.query.assert_called_once()
    tools[1].query_engine.query.assert_called_once()
    assert len(synth.synthesize.call_args.kwargs["nodes"]) == 1


def test_partial_failure_sync_use_async():
    """use_async=True via run_async_tasks: RuntimeError should not crash the query."""
    tools, question_gen, synth = _make_components()
    engine = SubQuestionQueryEngine(
        question_gen=question_gen,
        response_synthesizer=synth,
        query_engine_tools=tools,
        use_async=True,
        verbose=False,
    )

    response = engine.query(QueryBundle(query_str="compare sources"))

    assert str(response) == "final"
    tools[0].query_engine.aquery.assert_awaited_once()
    tools[1].query_engine.aquery.assert_awaited_once()
    assert len(synth.synthesize.call_args.kwargs["nodes"]) == 1


@pytest.mark.asyncio
async def test_partial_failure_aquery():
    """_aquery via asyncio.gather: RuntimeError should not crash the query."""
    tools, question_gen, synth = _make_components()
    engine = SubQuestionQueryEngine(
        question_gen=question_gen,
        response_synthesizer=synth,
        query_engine_tools=tools,
        use_async=True,
        verbose=False,
    )

    response = await engine.aquery(QueryBundle(query_str="compare sources"))

    assert str(response) == "final"
    tools[0].query_engine.aquery.assert_awaited_once()
    tools[1].query_engine.aquery.assert_awaited_once()
    assert len(synth.asynthesize.call_args.kwargs["nodes"]) == 1


def test_llm_question_generator_update_prompts_preserves_metadata_and_parser():
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.prompts.base import ChatPromptTemplate, PromptTemplate
    from llama_index.core.prompts.prompt_type import PromptType
    from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
    from llama_index.core.question_gen.output_parser import SubQuestionOutputParser

    class CustomParser(SubQuestionOutputParser):
        pass

    custom_parser = CustomParser()
    gen = LLMQuestionGenerator.from_defaults(
        llm=MockLLM(),
        output_parser=custom_parser,
    )

    new_prompt = PromptTemplate(
        "Tools: {tools_str}\nQuery: {query_str}\nPrefix: {prefix}",
        prompt_type=PromptType.SUB_QUESTION,
        function_mappings={"prefix": lambda **kwargs: "computed_prefix"},
    ).partial_format(extra="val")

    gen.update_prompts({"question_gen_prompt": new_prompt})
    updated = gen.get_prompts()["question_gen_prompt"]

    assert gen._prompt.output_parser is custom_parser
    assert isinstance(updated.output_parser, CustomParser)
    assert updated.metadata["prompt_type"] == PromptType.SUB_QUESTION
    assert updated.kwargs == {"extra": "val"}
    assert "prefix" in updated.function_mappings

    chat_prompt = ChatPromptTemplate.from_messages(
        [("user", "Tools: {tools_str}\nQuery: {query_str}")]
    )
    gen.update_prompts({"question_gen_prompt": chat_prompt})
    updated_chat = gen.get_prompts()["question_gen_prompt"]
    assert isinstance(updated_chat, ChatPromptTemplate)
    assert gen._prompt.output_parser is custom_parser
    assert isinstance(updated_chat.output_parser, CustomParser)


def test_sub_question_output_parser_single_object():
    from llama_index.core.question_gen.output_parser import SubQuestionOutputParser

    parser = SubQuestionOutputParser()
    raw = '```json\n{"sub_question": "What is revenue?", "tool_name": "uber_10k"}\n```'
    parsed = parser.parse(raw)
    assert parsed.parsed_output == [
        SubQuestion(sub_question="What is revenue?", tool_name="uber_10k")
    ]

