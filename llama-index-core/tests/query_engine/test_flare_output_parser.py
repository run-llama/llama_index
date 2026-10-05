from llama_index.core.query_engine.flare.output_parser import IsDoneOutputParser


def test_is_done_output_parser_uses_custom_predicate() -> None:
    parser = IsDoneOutputParser(
        is_done_fn=lambda output: output == "finished",
        fmt_answer_fn=lambda output: f"formatted: {output}",
    )

    assert parser.parse("finished") == (True, "formatted: finished")
