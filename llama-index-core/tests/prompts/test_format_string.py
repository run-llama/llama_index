from llama_index.core.prompts.utils import format_string


def test_formatter() -> None:
    """Test escape JSON.

    If there's curly brackets, escape it.

    """
    # create dumb test case
    test_case_1 = format_string("test message")
    assert test_case_1 == "test message"

    # create test case with two brackets
    test_case_2 = format_string("test {message} {test}", message="test", test="test")
    assert test_case_2 == "test test test"

    # create test case with a bracket that's already escaped - shouldn't change!
    test_case_3 = format_string("test {{message}} {test}", test="test")
    assert test_case_3 == "test {{message}} test"

    # test with json
    test_case_4 = format_string(
        """{"role": "user", "content": "test {message} {test}", "additional_kwargs": {"test": "test"}}""",
        message="test",
        test="test",
    )
    assert (
        test_case_4
        == """{"role": "user", "content": "test test test", "additional_kwargs": {"test": "test"}}"""
    )
