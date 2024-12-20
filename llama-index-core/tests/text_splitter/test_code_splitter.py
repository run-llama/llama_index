"""Test text splitter."""
import os
import pytest
from typing import List

try:
    import tree_sitter  # noqa
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
def test__py_custom_parser_code_splitter() -> None:
    """Test case for code splitting using custom parser generated from tree_sitter_languages."""
    if "CI" in os.environ:
        return

    from tree_sitter_languages import get_parser

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
