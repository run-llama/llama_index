from llama_index.core.llms.structured_llm import _escape_json
from llama_index.core.base.llms.types import ChatMessage
import json


def test_escape_json() -> None:
    """Test escape JSON.

    If there's curly brackets, escape it.

    """
    # create dumb test case
    test_case_1 = _escape_json([ChatMessage(role="user", content="test message")])
    assert test_case_1 == [ChatMessage(role="user", content="test message")]

    # create test case with two brackets
    test_case_2 = _escape_json(
        [ChatMessage(role="user", content="test {message} {test}")]
    )
    assert test_case_2 == [
        ChatMessage(role="user", content="test {{message}} {{test}}")
    ]

    # create test case with a bracket that's already escaped - shouldn't change!
    test_case_3 = _escape_json(
        [ChatMessage(role="user", content="test {{message}} {test}")]
    )
    print(test_case_3[0].content)
    assert test_case_3 == [
        ChatMessage(role="user", content="test {{message}} {{test}}")
    ]

    # test with additional kwargs
    test_case_4 = _escape_json(
        [
            ChatMessage(
                role="user",
                content="test {{message}} {test}",
                additional_kwargs={"test": "test"},
            )
        ]
    )
    assert test_case_4 == [
        ChatMessage(
            role="user",
            content="test {{message}} {{test}}",
            additional_kwargs={"test": "test"},
        )
    ]

    # shouldn't escape already escaped brackets with 4 brackets
    test_case_5 = _escape_json(
        [ChatMessage(role="user", content="test {{{{message}}}} {test}")]
    )
    assert test_case_5 == [
        ChatMessage(role="user", content="test {{{{message}}}} {{test}}")
    ]


    # test JSON
    test_json_str = json.dumps({"tests": [{"foo": "bar"}, {"baz": "{test}"}]})
    test_case_6 = _escape_json(
        [ChatMessage(role="user", content=test_json_str)]
    )
    print(test_case_6)
    assert test_case_6 == [
        ChatMessage(role="user", content="{{\"tests\": [{{\"foo\": \"bar\"}}, {{\"baz\": \"{{test}}\"}}]}}")
    ]
    raise Exception
        