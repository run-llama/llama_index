"""Tests for string manipulation utilities."""

from llama_index.tools.mcp_code_execution.strings import (
    clean_ansi,
    detect_dialog,
    detect_ipython_prompt,
    detect_prompt,
    truncate_output,
)


class TestCleanAnsi:
    def test_removes_color_codes(self) -> None:
        text = "\x1b[31mred text\x1b[0m"
        assert clean_ansi(text) == "red text"

    def test_removes_cursor_movement(self) -> None:
        text = "\x1b[2Jhello\x1b[H"
        assert clean_ansi(text) == "hello"

    def test_plain_text_unchanged(self) -> None:
        text = "hello world"
        assert clean_ansi(text) == "hello world"

    def test_empty_string(self) -> None:
        assert clean_ansi("") == ""

    def test_complex_escape_sequences(self) -> None:
        text = "\x1b[1;32mbold green\x1b[0m normal \x1b[4munderline\x1b[0m"
        assert clean_ansi(text) == "bold green normal underline"


class TestDetectPrompt:
    def test_dollar_prompt(self) -> None:
        assert detect_prompt("user@host:~$ ") is True

    def test_hash_prompt(self) -> None:
        assert detect_prompt("root@host:~# ") is True

    def test_python_repl(self) -> None:
        assert detect_prompt(">>> ") is True

    def test_ipython_prompt(self) -> None:
        assert detect_prompt("In [1]: ") is True

    def test_no_prompt(self) -> None:
        assert detect_prompt("some output text") is False

    def test_empty_string(self) -> None:
        assert detect_prompt("") is False

    def test_multiline_with_prompt_at_end(self) -> None:
        text = "some output\nmore output\nuser@host:~$ "
        assert detect_prompt(text) is True

    def test_venv_prompt(self) -> None:
        assert detect_prompt("(myenv) user@host:~$ ") is True


class TestDetectIPythonPrompt:
    def test_standard_ipython_prompt(self) -> None:
        assert detect_ipython_prompt("In [1]: ") is True

    def test_higher_number(self) -> None:
        assert detect_ipython_prompt("In [42]: ") is True

    def test_not_ipython(self) -> None:
        assert detect_ipython_prompt(">>> ") is False

    def test_empty(self) -> None:
        assert detect_ipython_prompt("") is False


class TestDetectDialog:
    def test_yes_no_bracket(self) -> None:
        assert detect_dialog("Do you want to proceed? [Y/n]") is True

    def test_yes_no_paren(self) -> None:
        assert detect_dialog("Continue? (yes/no)") is True

    def test_password_prompt(self) -> None:
        assert detect_dialog("Password: ") is True

    def test_are_you_sure(self) -> None:
        assert detect_dialog("Are you sure you want to delete?") is True

    def test_no_dialog(self) -> None:
        assert detect_dialog("normal output text") is False

    def test_empty(self) -> None:
        assert detect_dialog("") is False


class TestTruncateOutput:
    def test_short_text_unchanged(self) -> None:
        text = "short"
        assert truncate_output(text) == "short"

    def test_truncates_long_text(self) -> None:
        text = "x" * 60000
        result = truncate_output(text, max_chars=1000)
        assert len(result) < 60000
        assert "truncated" in result

    def test_preserves_start_and_end(self) -> None:
        text = "START" + "x" * 60000 + "END"
        result = truncate_output(text, max_chars=1000)
        assert result.startswith("START")
        assert result.endswith("END")

    def test_exact_limit_unchanged(self) -> None:
        text = "x" * 1000
        assert truncate_output(text, max_chars=1000) == text
