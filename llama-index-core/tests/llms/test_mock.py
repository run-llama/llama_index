from llama_index.core.llms import MockLLM


def test_mock_llm_stream_complete_empty_prompt_no_max_tokens() -> None:
    """
    Test that MockLLM.stream_complete with an empty prompt and max_tokens=None
    does not raise a validation error.
    This test case is based on issue #19353.
    """
    llm = MockLLM(max_tokens=None)
    response_gen = llm.stream_complete("")

    # Consume the generator to trigger the potential error
    responses = list(response_gen)

    # Check that we received a single, empty response
    assert len(responses) == 1
    assert responses[0].text == ""
    assert responses[0].delta == ""
