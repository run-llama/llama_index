"""Test text splitter."""
import os

from llama_index.langchain_helpers.text_splitter import (
    CodeSplitter,
    SentenceSplitter,
    TokenTextSplitter,
)


def test_split_token() -> None:
    """Test split normal token."""
    # tiktoken will say length is ~5k
    token = "foo bar"
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    chunks = text_splitter.split_text(token)
    assert chunks == ["foo", "bar"]

    token = "foo bar hello world"
    text_splitter = TokenTextSplitter(chunk_size=2, chunk_overlap=1)
    chunks = text_splitter.split_text(token)
    assert chunks == ["foo bar", "bar hello", "hello world"]


def test_truncate_token() -> None:
    """Test truncate normal token."""
    # tiktoken will say length is ~5k
    token = "foo bar"
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    chunks = text_splitter.truncate_text(token)
    assert chunks == "foo"


def test_split_long_token() -> None:
    """Test split a really long token."""
    # tiktoken will say length is ~5k
    token = "a" * 100
    text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    chunks = text_splitter.split_text(token)
    # each text chunk may have spaces, since we join splits by separator
    assert "".join(chunks).replace(" ", "") == token

    token = ("a" * 49) + "\n" + ("a" * 50)
    text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    chunks = text_splitter.split_text(token)
    assert len(chunks[0]) == 49
    assert len(chunks[1]) == 50


def test_split_with_metadata_str() -> None:
    """Test split while taking into account chunk size used by metadata str."""
    text = " ".join(["foo"] * 20)
    metadata_str = "test_metadata_str"

    text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    assert len(chunks) == 1

    text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    chunks = text_splitter.split_text(text, metadata_str=metadata_str)
    assert len(chunks) == 2


def test_split_diff_sentence_token() -> None:
    """Test case of a string that will split differently."""
    token_text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    sentence_text_splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)

    text = " ".join(["foo"] * 15) + "\n\n\n" + " ".join(["bar"] * 15)
    token_split = token_text_splitter.split_text(text)
    sentence_split = sentence_text_splitter.split_text(text)
    assert token_split[0] == " ".join(["foo"] * 15) + "\n\n\n" + " ".join(["bar"] * 3)
    assert token_split[1] == " ".join(["bar"] * 12)
    assert sentence_split[0] == " ".join(["foo"] * 15)
    assert sentence_split[1] == " ".join(["bar"] * 15)


def test_split_diff_sentence_token2() -> None:
    """Test case of a string that will split differently."""
    token_text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    sentence_text_splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)

    text = " ".join(["foo"] * 15) + ". " + " ".join(["bar"] * 15)
    token_split = token_text_splitter.split_text(text)
    sentence_split = sentence_text_splitter.split_text(text)

    assert token_split[0] == " ".join(["foo"] * 15) + ". " + " ".join(["bar"] * 4)
    assert token_split[1] == " ".join(["bar"] * 11)
    assert sentence_split[0] == " ".join(["foo"] * 15) + "."
    assert sentence_split[1] == " ".join(["bar"] * 15)


def test_python_code_splitter() -> None:
    """Test case for code splitting using python"""

    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="python", chunk_lines=4, chunk_lines_overlap=1, max_chars=30
    )

    text = """\
def foo():
    print("bar")

def baz():
    print("bbq")"""

    chunks = code_splitter.split_text(text)
    assert chunks[0].startswith("def foo():")
    assert chunks[1].startswith("def baz():")


def test_typescript_code_splitter() -> None:
    """Test case for code splitting using typescript"""

    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="typescript", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
    )

    text = """\
function foo() {
    console.log("bar");
}

function baz() {
    console.log("bbq");
}"""

    chunks = code_splitter.split_text(text, "typescript")
    assert chunks[0].startswith("function foo()")
    assert chunks[1].startswith("function baz()")


# def test_autolang_code_splitter() -> None:
#     """Test case for code splitting"""
#     code_splitter = CodeSplitter(chunk_lines=4, chunk_lines_overlap=1, max_chars=40)

#     text = """\
# function foo() {
#     print("bar")
# }

# function baz() {
#     print("bbq")
# }"""

#     chunks = code_splitter.split_text(text)
#     assert chunks[0].startswith("function foo()")
#     assert chunks[1].startswith("function baz()")

#     code_splitter = CodeSplitter(chunk_lines=4, chunk_lines_overlap=1, max_chars=40)

#     text = """\
# void foo() {
#     std::cout << "foo" << std::endl;
# }

# void baz() {
#     std::cout << "bbq" << std::endl;
# }"""

#     chunks = code_splitter.split_text(text)
#     assert chunks[0].startswith("void foo()")
#     assert chunks[1].startswith("void baz()")
