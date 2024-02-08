from typing import TYPE_CHECKING, Any, Union

import pytest
from llama_index.legacy.llms import ChatMessage
from pytest import MonkeyPatch

if TYPE_CHECKING:
    from ai21.ai21_object import AI21Object

try:
    import ai21
    from ai21.ai21_object import construct_ai21_object
except ImportError:
    ai21 = None  # type: ignore


from llama_index.legacy.llms.ai21 import AI21


def mock_completion(*args: Any, **kwargs: Any) -> Union[Any, "AI21Object"]:
    return construct_ai21_object(
        {
            "id": "f6adacef-0e94-6353-244f-df8d38954b19",
            "prompt": {
                "text": "This is just a test",
                "tokens": [
                    {
                        "generatedToken": {
                            "token": "▁This▁is▁just",
                            "logprob": -13.657383918762207,
                            "raw_logprob": -13.657383918762207,
                        },
                        "topTokens": None,
                        "textRange": {"start": 0, "end": 12},
                    },
                    {
                        "generatedToken": {
                            "token": "▁a▁test",
                            "logprob": -4.080351829528809,
                            "raw_logprob": -4.080351829528809,
                        },
                        "topTokens": None,
                        "textRange": {"start": 12, "end": 19},
                    },
                ],
            },
            "completions": [
                {
                    "data": {
                        "text": "\nThis is a test to see if my text is showing up correctly.",
                        "tokens": [
                            {
                                "generatedToken": {
                                    "token": "<|newline|>",
                                    "logprob": 0,
                                    "raw_logprob": -0.01992332935333252,
                                },
                                "topTokens": None,
                                "textRange": {"start": 0, "end": 1},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁This▁is▁a",
                                    "logprob": -0.00014733182615600526,
                                    "raw_logprob": -1.228371500968933,
                                },
                                "topTokens": None,
                                "textRange": {"start": 1, "end": 10},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁test",
                                    "logprob": 0,
                                    "raw_logprob": -0.0422857291996479,
                                },
                                "topTokens": None,
                                "textRange": {"start": 10, "end": 15},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁to▁see▁if",
                                    "logprob": -0.4861462712287903,
                                    "raw_logprob": -1.2263909578323364,
                                },
                                "topTokens": None,
                                "textRange": {"start": 15, "end": 25},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁my",
                                    "logprob": -9.536738616588991e-7,
                                    "raw_logprob": -0.8164164423942566,
                                },
                                "topTokens": None,
                                "textRange": {"start": 25, "end": 28},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁text",
                                    "logprob": -0.003087161108851433,
                                    "raw_logprob": -1.7130306959152222,
                                },
                                "topTokens": None,
                                "textRange": {"start": 28, "end": 33},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁is",
                                    "logprob": -1.8836627006530762,
                                    "raw_logprob": -0.9880049824714661,
                                },
                                "topTokens": None,
                                "textRange": {"start": 33, "end": 36},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁showing▁up",
                                    "logprob": -0.00006341733387671411,
                                    "raw_logprob": -0.954255223274231,
                                },
                                "topTokens": None,
                                "textRange": {"start": 36, "end": 47},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁correctly",
                                    "logprob": -0.00022098960471339524,
                                    "raw_logprob": -0.6004139184951782,
                                },
                                "topTokens": None,
                                "textRange": {"start": 47, "end": 57},
                            },
                            {
                                "generatedToken": {
                                    "token": ".",
                                    "logprob": 0,
                                    "raw_logprob": -0.039214372634887695,
                                },
                                "topTokens": None,
                                "textRange": {"start": 57, "end": 58},
                            },
                            {
                                "generatedToken": {
                                    "token": "<|endoftext|>",
                                    "logprob": 0,
                                    "raw_logprob": -0.22456447780132294,
                                },
                                "topTokens": None,
                                "textRange": {"start": 58, "end": 58},
                            },
                        ],
                    },
                    "finishReason": {"reason": "endoftext"},
                }
            ],
        }
    )


def mock_chat(*args: Any, **kwargs: Any) -> Union[Any, "AI21Object"]:
    return construct_ai21_object(
        {
            "id": "f8d0cd0a-7c85-deb2-16b3-491c7ffdd4f2",
            "prompt": {
                "text": "user: This is just a test assistant:",
                "tokens": [
                    {
                        "generatedToken": {
                            "token": "▁user",
                            "logprob": -13.633946418762207,
                            "raw_logprob": -13.633946418762207,
                        },
                        "topTokens": None,
                        "textRange": {"start": 0, "end": 4},
                    },
                    {
                        "generatedToken": {
                            "token": ":",
                            "logprob": -5.545032978057861,
                            "raw_logprob": -5.545032978057861,
                        },
                        "topTokens": None,
                        "textRange": {"start": 4, "end": 5},
                    },
                    {
                        "generatedToken": {
                            "token": "▁This▁is▁just",
                            "logprob": -10.848762512207031,
                            "raw_logprob": -10.848762512207031,
                        },
                        "topTokens": None,
                        "textRange": {"start": 5, "end": 18},
                    },
                    {
                        "generatedToken": {
                            "token": "▁a▁test",
                            "logprob": -2.0551252365112305,
                            "raw_logprob": -2.0551252365112305,
                        },
                        "topTokens": None,
                        "textRange": {"start": 18, "end": 25},
                    },
                    {
                        "generatedToken": {
                            "token": "▁assistant",
                            "logprob": -17.020610809326172,
                            "raw_logprob": -17.020610809326172,
                        },
                        "topTokens": None,
                        "textRange": {"start": 25, "end": 35},
                    },
                    {
                        "generatedToken": {
                            "token": ":",
                            "logprob": -12.311965942382812,
                            "raw_logprob": -12.311965942382812,
                        },
                        "topTokens": None,
                        "textRange": {"start": 35, "end": 36},
                    },
                ],
            },
            "completions": [
                {
                    "data": {
                        "text": "\nassistant:\nHow can I assist you today?",
                        "tokens": [
                            {
                                "generatedToken": {
                                    "token": "<|newline|>",
                                    "logprob": 0,
                                    "raw_logprob": -0.02031332440674305,
                                },
                                "topTokens": None,
                                "textRange": {"start": 0, "end": 1},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁assistant",
                                    "logprob": 0,
                                    "raw_logprob": -0.24520651996135712,
                                },
                                "topTokens": None,
                                "textRange": {"start": 1, "end": 10},
                            },
                            {
                                "generatedToken": {
                                    "token": ":",
                                    "logprob": 0,
                                    "raw_logprob": -0.0026112052146345377,
                                },
                                "topTokens": None,
                                "textRange": {"start": 10, "end": 11},
                            },
                            {
                                "generatedToken": {
                                    "token": "<|newline|>",
                                    "logprob": 0,
                                    "raw_logprob": -0.3382393717765808,
                                },
                                "topTokens": None,
                                "textRange": {"start": 11, "end": 12},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁How▁can▁I",
                                    "logprob": -0.000008106198947643861,
                                    "raw_logprob": -1.3073582649230957,
                                },
                                "topTokens": None,
                                "textRange": {"start": 12, "end": 21},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁assist▁you",
                                    "logprob": -2.15450382232666,
                                    "raw_logprob": -0.8163930177688599,
                                },
                                "topTokens": None,
                                "textRange": {"start": 21, "end": 32},
                            },
                            {
                                "generatedToken": {
                                    "token": "▁today",
                                    "logprob": 0,
                                    "raw_logprob": -0.1474292278289795,
                                },
                                "topTokens": None,
                                "textRange": {"start": 32, "end": 38},
                            },
                            {
                                "generatedToken": {
                                    "token": "?",
                                    "logprob": 0,
                                    "raw_logprob": -0.011986607685685158,
                                },
                                "topTokens": None,
                                "textRange": {"start": 38, "end": 39},
                            },
                            {
                                "generatedToken": {
                                    "token": "<|endoftext|>",
                                    "logprob": -1.1920928244535389e-7,
                                    "raw_logprob": -0.2295214682817459,
                                },
                                "topTokens": None,
                                "textRange": {"start": 39, "end": 39},
                            },
                        ],
                    },
                    "finishReason": {"reason": "endoftext"},
                }
            ],
        }
    )


@pytest.mark.skipif(ai21 is None, reason="ai21 not installed")
def test_completion_model_basic(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("ai21.Completion.execute", mock_completion)

    mock_api_key = "fake_key"
    llm = AI21(model="j2-mid", api_key=mock_api_key)

    test_prompt = "This is just a test"
    response = llm.complete(test_prompt)
    assert (
        response.text == "\nThis is a test to see if my text is showing up correctly."
    )

    monkeypatch.setattr("ai21.Completion.execute", mock_chat)

    message = ChatMessage(role="user", content=test_prompt)
    chat_response = llm.chat([message])
    print(chat_response.message.content)
    assert chat_response.message.content == "\nassistant:\nHow can I assist you today?"
