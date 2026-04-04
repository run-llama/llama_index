"""Regression tests for _separate_thinking (fixes #20456)."""

from types import SimpleNamespace

from llama_index.llms.mistralai import MistralAI


class FakeTextChunk:
    def __init__(self, text: str = "") -> None:
        self.text = text


class FakeThinkChunk:
    def __init__(self, thinking: list = None) -> None:  # type: ignore[assignment]
        self.thinking = thinking or []


fake_models = SimpleNamespace(
    TextChunk=FakeTextChunk,
    ThinkChunk=FakeThinkChunk,
)


def _make_llm() -> MistralAI:
    llm = MistralAI(api_key="fake")
    llm._models = fake_models
    return llm


def test_structured_chunks_basic() -> None:
    llm = _make_llm()
    chunks = [
        FakeThinkChunk(thinking=[FakeTextChunk(text="reasoning")]),
        FakeTextChunk(text="Paris is the capital"),
    ]
    thinking, response = llm._separate_thinking(chunks)
    assert thinking == "reasoning"
    assert response == "Paris is the capital"


def test_structured_no_thinking() -> None:
    llm = _make_llm()
    chunks = [FakeTextChunk(text="just a response")]
    thinking, response = llm._separate_thinking(chunks)
    assert thinking == ""
    assert response == "just a response"


def test_structured_no_response() -> None:
    llm = _make_llm()
    chunks = [FakeThinkChunk(thinking=[FakeTextChunk(text="only thinking")])]
    thinking, response = llm._separate_thinking(chunks)
    assert thinking == "only thinking"
    assert response == ""


def test_structured_multiple_chunks() -> None:
    llm = _make_llm()
    chunks = [
        FakeThinkChunk(thinking=[FakeTextChunk(text="step1")]),
        FakeThinkChunk(thinking=[FakeTextChunk(text="step2")]),
        FakeTextChunk(text="answer"),
    ]
    thinking, response = llm._separate_thinking(chunks)
    assert "step1" in thinking
    assert "step2" in thinking
    assert response == "answer"


def test_string_with_tags() -> None:
    llm = MistralAI(api_key="fake")
    thinking, response = llm._separate_thinking(
        "<think>\nreasoning\n</think>\nresponse"
    )
    assert thinking == "reasoning"
    assert "response" in response


def test_string_no_tags() -> None:
    llm = MistralAI(api_key="fake")
    thinking, response = llm._separate_thinking("just a response")
    assert thinking == ""
    assert response == "just a response"


def test_empty_list() -> None:
    llm = _make_llm()
    thinking, response = llm._separate_thinking([])
    assert thinking == ""
    assert response == ""
