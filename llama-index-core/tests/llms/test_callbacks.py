import pytest
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.mock import MockLLM


@pytest.fixture()
def llm() -> LLM:
    return MockLLM()


@pytest.fixture()
def prompt() -> str:
    return "test prompt"


def test_llm_complete_prompt_arg(llm: LLM, prompt: str) -> None:
    res = llm.complete(prompt)
    expected_res_text = prompt
    assert res.text == expected_res_text


def test_llm_complete_prompt_kwarg(llm: LLM, prompt: str) -> None:
    res = llm.complete(prompt=prompt)
    expected_res_text = prompt
    assert res.text == expected_res_text


def test_llm_complete_throws_if_duplicate_prompt(llm: LLM, prompt: str) -> None:
    with pytest.raises(TypeError):
        llm.complete(prompt, prompt=prompt)


def test_llm_complete_throws_if_no_prompt(llm: LLM) -> None:
    with pytest.raises(ValueError):
        llm.complete()


def test_llm_stream_complete_prompt_arg(llm: LLM, prompt: str) -> None:
    res_text = "".join(r.delta for r in llm.stream_complete(prompt))
    expected_res_text = prompt
    assert res_text == expected_res_text


def test_llm_stream_complete_prompt_kwarg(llm: LLM, prompt: str) -> None:
    res_text = "".join(r.delta for r in llm.stream_complete(prompt=prompt))
    expected_res_text = prompt
    assert res_text == expected_res_text


def test_llm_stream_complete_throws_if_duplicate_prompt(llm: LLM, prompt: str) -> None:
    with pytest.raises(TypeError):
        llm.stream_complete(prompt, prompt=prompt)


def test_llm_stream_complete_throws_if_no_prompt(llm: LLM) -> None:
    with pytest.raises(ValueError):
        llm.stream_complete()
