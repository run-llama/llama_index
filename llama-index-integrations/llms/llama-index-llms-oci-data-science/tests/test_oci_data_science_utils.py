import pytest
from llama_index.core.base.llms.types import ChatMessage, LogProb, MessageRole
from llama_index.llms.oci_data_science.utils import (
    UnsupportedOracleAdsVersionError,
    _from_completion_logprobs_dict,
    _from_message_dict,
    _from_token_logprob_dicts,
    _get_response_token_counts,
    _resolve_tool_choice,
    _to_message_dicts,
    _update_tool_calls,
    # _validate_dependency,
)


class TestUnsupportedOracleAdsVersionError:
    """Unit tests for UnsupportedOracleAdsVersionError."""

    def test_exception_message(self):
        """Ensures the exception message is formatted correctly."""
        current_version = "2.12.5"
        required_version = "2.12.6"
        expected_message = (
            f"The `oracle-ads` version {current_version} currently installed is incompatible with "
            "the `llama-index-llms-oci-data-science` version in use. To resolve this issue, "
            f"please upgrade to `oracle-ads:{required_version}` or later using the "
            "command: `pip install oracle-ads -U`"
        )

        exception = UnsupportedOracleAdsVersionError(current_version, required_version)
        assert str(exception) == expected_message


# class TestValidateDependency:
#     """Unit tests for _validate_dependency decorator."""

#     def setup_method(self):
#         @_validate_dependency
#         def sample_function():
#             return "function executed"

#         self.sample_function = sample_function

#     @patch("llama_index.llms.oci_data_science.utils.MIN_ADS_VERSION", new="2.12.6")
#     @patch("ads.__version__", new="2.12.7")
#     def test_valid_version(self):
#         """Ensures the function executes when the oracle-ads version is sufficient."""
#         result = self.sample_function()
#         assert result == "function executed"

#     @patch("llama_index.llms.oci_data_science.utils.MIN_ADS_VERSION", new="2.12.6")
#     @patch("ads.__version__", new="2.12.5")
#     def test_unsupported_version(self):
#         """Ensures UnsupportedOracleAdsVersionError is raised for insufficient version."""
#         with pytest.raises(UnsupportedOracleAdsVersionError) as exc_info:
#             self.sample_function()

#     @patch("llama_index.llms.oci_data_science.utils.MIN_ADS_VERSION", new="2.12.6")
#     def test_oracle_ads_not_installed(self):
#         """Ensures ImportError is raised when oracle-ads is not installed."""
#         with patch.dict("sys.modules", {"ads": None}):
#             with pytest.raises(ImportError) as exc_info:
#                 self.sample_function()
#             assert "Could not import `oracle-ads` Python package." in str(
#                 exc_info.value
#             )


class TestToMessageDicts:
    """Unit tests for _to_message_dicts function."""

    def test_sequence_conversion(self):
        """Ensures a sequence of ChatMessages is converted correctly."""
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]
        expected_result = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _to_message_dicts(messages)
        assert result == expected_result

    def test_empty_sequence(self):
        """Ensures the function works with an empty sequence."""
        messages = []
        expected_result = []
        result = _to_message_dicts(messages)
        assert result == expected_result

    def test_drop_none(self):
        """Ensures drop_none parameter works correctly for sequences."""
        messages = [
            ChatMessage(role=MessageRole.USER, content=None),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Hi there!",
                additional_kwargs={"custom_field": None},
            ),
        ]
        expected_result = [
            {"role": "user"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _to_message_dicts(messages, drop_none=True)
        assert result == expected_result


class TestFromCompletionLogprobs:
    """Unit tests for _from_completion_logprobs_dict function."""

    def test_conversion(self):
        """Ensures  completion logprobs are converted correctly."""
        logprobs = {
            "tokens": ["Hello", "world"],
            "token_logprobs": [-0.1, -0.2],
            "top_logprobs": [
                {"Hello": -0.1, "Hi": -1.0},
                {"world": -0.2, "earth": -1.2},
            ],
        }
        expected_result = [
            [
                LogProb(token="Hello", logprob=-0.1, bytes=[]),
                LogProb(token="Hi", logprob=-1.0, bytes=[]),
            ],
            [
                LogProb(token="world", logprob=-0.2, bytes=[]),
                LogProb(token="earth", logprob=-1.2, bytes=[]),
            ],
        ]
        result = _from_completion_logprobs_dict(logprobs)
        assert result == expected_result

    def test_empty_logprobs(self):
        """Ensures function returns empty list when no logprobs are provided."""
        logprobs = {}
        expected_result = []
        result = _from_completion_logprobs_dict(logprobs)
        assert result == expected_result


class TestFromTokenLogprobs:
    """Unit tests for _from_token_logprob_dicts function."""

    def test_conversion(self):
        """Ensures multiple  token logprobs are converted correctly."""
        token_logprob_dicts = [
            {
                "token": "Hello",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "Hello", "logprob": -0.1, "bytes": [1, 2, 3]},
                    {"token": "Hi", "logprob": -1.0, "bytes": [1, 2, 3]},
                ],
            },
            {
                "token": "world",
                "logprob": -0.2,
                "top_logprobs": [
                    {"token": "world", "logprob": -0.2, "bytes": [2, 3, 4]},
                    {"token": "earth", "logprob": -1.2, "bytes": [2, 3, 4]},
                ],
            },
        ]
        expected_result = [
            [
                LogProb(token="Hello", logprob=-0.1, bytes=[1, 2, 3]),
                LogProb(token="Hi", logprob=-1.0, bytes=[1, 2, 3]),
            ],
            [
                LogProb(token="world", logprob=-0.2, bytes=[2, 3, 4]),
                LogProb(token="earth", logprob=-1.2, bytes=[2, 3, 4]),
            ],
        ]
        result = _from_token_logprob_dicts(token_logprob_dicts)
        assert result == expected_result

    def test_empty_input(self):
        """Ensures function returns empty list when input is empty."""
        token_logprob_dicts = []
        expected_result = []
        result = _from_token_logprob_dicts(token_logprob_dicts)
        assert result == expected_result


class TestFromMessage:
    """Unit tests for _from_message_dict function."""

    def test_conversion(self):
        """Ensures an  message dict is converted to ChatMessage."""
        message_dict = {
            "role": "assistant",
            "content": "Hello!",
            "tool_calls": [{"name": "tool1", "arguments": "arg1"}],
        }
        expected_result = ChatMessage(
            role="assistant",
            content="Hello!",
            additional_kwargs={"tool_calls": [{"name": "tool1", "arguments": "arg1"}]},
        )
        result = _from_message_dict(message_dict)
        assert result == expected_result

    def test_missing_optional_fields(self):
        """Ensures function works when optional fields are missing."""
        message_dict = {"role": "user", "content": "Hi!"}
        expected_result = ChatMessage(
            role="user", content="Hi!", additional_kwargs={"tool_calls": []}
        )
        result = _from_message_dict(message_dict)
        assert result == expected_result


class TestGetResponseTokenCounts:
    """Unit tests for _get_response_token_counts function."""

    def test_with_usage(self):
        """Ensures token counts are extracted correctly when usage is present."""
        raw_response = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        }
        expected_result = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        result = _get_response_token_counts(raw_response)
        assert result == expected_result

    def test_without_usage(self):
        """Ensures function returns empty dict when usage is missing."""
        raw_response = {}
        expected_result = {}
        result = _get_response_token_counts(raw_response)
        assert result == expected_result

    def test_missing_token_counts(self):
        """Ensures missing token counts default to zero."""
        raw_response = {"usage": {}}
        result = _get_response_token_counts(raw_response)
        assert result == {}

        raw_response = {"usage": {"prompt_tokens": 10}}
        expected_result = {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        result = _get_response_token_counts(raw_response)
        assert result == expected_result


class TestUpdateToolCalls:
    """Unit tests for _update_tool_calls function."""

    def test_add_new_call(self):
        """Ensures a new tool call is added when indices do not match."""
        tool_calls = [{"index": 0, "function": {"name": "tool1", "arguments": "arg1"}}]
        tool_calls_delta = [
            {"index": 1, "function": {"name": "tool2", "arguments": "arg2"}}
        ]
        expected_result = [
            {"index": 0, "function": {"name": "tool1", "arguments": "arg1"}},
            {"index": 1, "function": {"name": "tool2", "arguments": "arg2"}},
        ]
        result = _update_tool_calls(tool_calls, tool_calls_delta)
        assert result == expected_result

    def test_update_existing_call(self):
        """Ensures the existing tool call is updated when indices match."""
        tool_calls = [{"index": 0, "function": {"name": "tool", "arguments": "arg"}}]
        tool_calls_delta = [{"index": 0, "function": {"name": "1", "arguments": "1"}}]
        expected_result = [
            {
                "index": 0,
                "function": {"name": "tool1", "arguments": "arg1"},
                "id": "",
            }
        ]
        result = _update_tool_calls(tool_calls, tool_calls_delta)
        assert result[0]["function"]["name"] == "tool1"
        assert result[0]["function"]["arguments"] == "arg1"

    def test_no_delta(self):
        """Ensures the original tool_calls is returned when delta is None."""
        tool_calls = [{"index": 0, "function": {"name": "tool1", "arguments": "arg1"}}]
        tool_calls_delta = None
        expected_result = [
            {"index": 0, "function": {"name": "tool1", "arguments": "arg1"}}
        ]
        result = _update_tool_calls(tool_calls, tool_calls_delta)
        assert result == expected_result

    def test_empty_tool_calls(self):
        """Ensures tool_calls is initialized when empty."""
        tool_calls = []
        tool_calls_delta = [
            {"index": 0, "function": {"name": "tool1", "arguments": "arg1"}}
        ]
        expected_result = [
            {"index": 0, "function": {"name": "tool1", "arguments": "arg1"}}
        ]
        result = _update_tool_calls(tool_calls, tool_calls_delta)
        assert result == expected_result


class TestResolveToolChoice:
    """Unit tests for _resolve_tool_choice function."""

    @pytest.mark.parametrize(
        ("input_choice", "expected_output"),
        [
            ("auto", "auto"),
            ("none", "none"),
            ("required", "required"),
            (
                "custom_tool",
                {"type": "function", "function": {"name": "custom_tool"}},
            ),
            (
                {"type": "function", "function": {"name": "custom_tool"}},
                {"type": "function", "function": {"name": "custom_tool"}},
            ),
        ],
    )
    def test_resolve_tool_choice(self, input_choice, expected_output):
        """Ensures tool choices are resolved correctly."""
        result = _resolve_tool_choice(input_choice)
        assert result == expected_output
