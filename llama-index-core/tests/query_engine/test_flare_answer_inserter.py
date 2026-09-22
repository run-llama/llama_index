from llama_index.core.query_engine.flare.answer_inserter import (
    DirectLookaheadAnswerInserter,
)
from llama_index.core.query_engine.flare.output_parser import QueryTaskOutputParser


def test_direct_inserter_multiple_tasks() -> None:
    response = (
        "Red is for [Search(why is red on the flag?)], green for "
        "[Search(why is green on the flag?)], and gold for mineral wealth."
    )
    tasks = QueryTaskOutputParser().parse(response)
    answers = [
        "the blood of those who died for independence",
        "the forests and farms",
    ]

    out = DirectLookaheadAnswerInserter().insert(response, tasks, answers)

    assert out == (
        "Red is for the blood of those who died for independence, green for "
        "the forests and farms, and gold for mineral wealth."
    )


def test_direct_inserter_single_task() -> None:
    response = "Red is for [Search(why is red on the flag?)]."
    tasks = QueryTaskOutputParser().parse(response)

    out = DirectLookaheadAnswerInserter().insert(response, tasks, ["courage"])

    assert out == "Red is for courage."
