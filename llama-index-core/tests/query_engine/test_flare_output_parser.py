from llama_index.core.query_engine.flare.output_parser import QueryTaskOutputParser


def test_query_task_parser_extracts_search_query() -> None:
    parser = QueryTaskOutputParser()
    tasks = parser.parse("Look this up [Search(cats)] please")
    assert len(tasks) == 1
    assert tasks[0].query_str == "cats"


def test_query_task_parser_skips_brackets_without_call() -> None:
    parser = QueryTaskOutputParser()
    tasks = parser.parse("Use [citation] then [Search(dogs)]")
    assert len(tasks) == 1
    assert tasks[0].query_str == "dogs"


def test_query_task_parser_handles_empty_output() -> None:
    parser = QueryTaskOutputParser()
    assert parser.parse("") == []
    assert parser.parse(None) == []
