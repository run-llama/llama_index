"""Test text splitter."""

import os
import pytest
from typing import List
from pydantic import ValidationError

try:
    import tree_sitter_language_pack  # noqa
    from llama_index.core.text_splitter import CodeSplitter

    SHOULD_SKIP = False
except ImportError:
    SHOULD_SKIP = True


from llama_index.core.schema import Document, MetadataMode, TextNode


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_python_code_splitter() -> None:
    """Test case for code splitting using python."""
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


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_start_end_char_idx() -> None:
    text = """\
def foo():
    print("bar")

def baz():
    print("bbq")"""
    document = Document(text=text)
    code_splitter = CodeSplitter(
        language="python", chunk_lines=4, chunk_lines_overlap=1, max_chars=30
    )
    nodes: List[TextNode] = code_splitter.get_nodes_from_documents([document])
    for node in nodes:
        assert node.start_char_idx is not None
        assert node.end_char_idx is not None
        assert node.end_char_idx - node.start_char_idx == len(
            node.get_content(metadata_mode=MetadataMode.NONE)
        )


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_typescript_code_splitter() -> None:
    """Test case for code splitting using typescript."""
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

    chunks = code_splitter.split_text(text)
    assert chunks[0].startswith("function foo()")
    assert chunks[1].startswith("function baz()")


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_html_code_splitter() -> None:
    """Test case for code splitting using typescript."""
    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="html", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
    )

    text = """\
<!DOCTYPE html>
<html>
<head>
    <title>My Example Page</title>
</head>
<body>
    <h1>Welcome to My Example Page</h1>
    <p>This is a basic HTML page example.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
    <img src="https://example.com/image.jpg" alt="Example Image">
</body>
</html>"""

    chunks = code_splitter.split_text(text)
    assert chunks[0].startswith("<!DOCTYPE html>")
    assert chunks[1].startswith("<html>")
    assert chunks[2].startswith("<head>")


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_tsx_code_splitter() -> None:
    """Test case for code splitting using typescript."""
    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="typescript", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
    )

    text = """\
import React from 'react';

interface Person {
  name: string;
  age: number;
}

const ExampleComponent: React.FC = () => {
  const person: Person = {
    name: 'John Doe',
    age: 30,
  };

  return (
    <div>
      <h1>Hello, {person.name}!</h1>
      <p>You are {person.age} years old.</p>
    </div>
  );
};

export default ExampleComponent;"""

    chunks = code_splitter.split_text(text)
    assert chunks[0].startswith("import React from 'react';")
    assert chunks[1].startswith("interface Person")


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_cpp_code_splitter() -> None:
    """Test case for code splitting using typescript."""
    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="cpp", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
    )

    text = """\
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}"""

    chunks = code_splitter.split_text(text)
    assert chunks[0].startswith("#include <iostream>")
    assert chunks[1].startswith("int main()")
    assert chunks[2].startswith("{\n    std::cout")


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_java_code_splitter() -> None:
    """Test case for code splitting using java."""
    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="java", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
    )

    text = """\
public static void foo() {
    System.out.println("bar");
}

public static void baz() {
    System.out.println("bbq");
}"""

    chunks = code_splitter.split_text(text)
    assert chunks[0].startswith("public static void foo()")
    assert chunks[2].startswith("public static void baz()")


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test__py_custom_parser_code_splitter() -> None:
    """Test case for code splitting using custom parser generated from tree_sitter_language_pack."""
    if "CI" in os.environ:
        return

    from tree_sitter_language_pack import get_parser

    parser = get_parser("python")

    code_splitter = CodeSplitter(
        language="custom",
        chunk_lines=4,
        chunk_lines_overlap=1,
        max_chars=30,
        parser=parser,
    )

    text = """\
def foo():
    print("bar")

def baz():
    print("bbq")"""

    chunks = code_splitter.split_text(text)
    assert chunks[0].startswith("def foo():")
    assert chunks[1].startswith("def baz():")


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_token_based_code_splitter() -> None:
    """Test case for token-based code splitting using python."""
    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="python",
        count_mode="token",
        max_tokens=20,  # Small token limit for testing
    )

    text = """\
def foo():
    print("bar")
    print("another line")

def baz():
    print("bbq")
    return "result"

def another_function():
    x = 1 + 2 + 3
    return x"""

    chunks = code_splitter.split_text(text)

    # Should create multiple chunks due to token limit
    assert len(chunks) > 1
    assert chunks[0].startswith("def foo():")
    assert "def baz():" in chunks[1] or "def another_function():" in chunks[1]


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_char_vs_token_mode_comparison() -> None:
    """Test comparison between character and token modes."""
    if "CI" in os.environ:
        return

    text = """\
def calculate_complex_result():
    result = some_very_long_variable_name + another_long_variable_name
    return result

def process_data():
    data = {"key1": "value1", "key2": "value2"}
    return data"""

    # Character-based splitter
    char_splitter = CodeSplitter(language="python", count_mode="char", max_chars=50)

    # Token-based splitter
    token_splitter = CodeSplitter(language="python", count_mode="token", max_tokens=15)

    char_chunks = char_splitter.split_text(text)
    token_chunks = token_splitter.split_text(text)

    # Both should split the text, but potentially differently
    assert len(char_chunks) >= 1
    assert len(token_chunks) >= 1

    # Verify chunks are not empty
    assert all(len(chunk.strip()) > 0 for chunk in char_chunks)
    assert all(len(chunk.strip()) > 0 for chunk in token_chunks)


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_backwards_compatibility() -> None:
    """Test that existing character-based functionality still works."""
    if "CI" in os.environ:
        return

    # Test with old parameters (should default to character mode)
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

    # Verify it's using character mode by default
    assert code_splitter.count_mode == "char"


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_custom_tokenizer() -> None:
    """Test using a custom tokenizer function."""
    if "CI" in os.environ:
        return

    # Simple custom tokenizer that splits on spaces
    def simple_tokenizer(text: str) -> List[str]:
        return text.split()

    code_splitter = CodeSplitter(
        language="python",
        count_mode="token",
        max_tokens=5,  # Very small limit for testing
        tokenizer=simple_tokenizer,
    )

    text = """\
def foo():
    print("bar")"""

    chunks = code_splitter.split_text(text)
    assert len(chunks) >= 1
    assert chunks[0].startswith("def foo():")


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_invalid_count_mode() -> None:
    """Test that invalid count_mode raises ValidationError."""
    if "CI" in os.environ:
        return

    with pytest.raises(ValidationError):
        CodeSplitter(language="python", count_mode="invalid_mode")


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_from_defaults_with_token_mode() -> None:
    """Test from_defaults class method with token mode parameters."""
    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter.from_defaults(
        language="python", count_mode="token", max_tokens=25
    )

    text = """\
def example_function():
    result = calculate_something()
    return result"""

    chunks = code_splitter.split_text(text)
    assert len(chunks) >= 1
    assert chunks[0].startswith("def example_function():")
    assert code_splitter.count_mode == "token"
    assert code_splitter.max_tokens == 25


@pytest.mark.skipif(SHOULD_SKIP, reason="tree_sitter not installed")
def test_token_mode_node_parsing() -> None:
    """Test token-based mode with node parsing."""
    if "CI" in os.environ:
        return

    text = """\
def complex_function():
    # This is a comment
    variable_with_very_long_name = "some string value"
    another_variable = variable_with_very_long_name.upper()
    return another_variable"""

    document = Document(text=text)
    code_splitter = CodeSplitter(language="python", count_mode="token", max_tokens=20)

    nodes = code_splitter.get_nodes_from_documents([document])

    assert len(nodes) >= 1
    assert isinstance(nodes[0], TextNode)
    assert "def complex_function():" in nodes[0].text
