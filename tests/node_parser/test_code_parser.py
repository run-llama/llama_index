"""Test text splitter."""
import os
from typing import List

from llama_index.node_parser.code_hierarchy import CodeBlockNodeParser
from llama_index.node_parser.interface import NodeParser
from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode

def test_python_code_splitter() -> None:
    """Test case for code splitting using python"""

    if "CI" in os.environ:
        return

    code_splitter = CodeBlockNodeParser(
        language="python",
        split_on_types=["FunctionDef", "ClassDef"],
    )

    text = """\
class Foo:
    def foo():
        print("bar")

    def baz():
        print("baz")
        """

    text_node = TextNode(
        text=text,
        metadata={
            'module': 'example.foo',
        }
    )

    chunks: List[RelatedNodeInfo] = code_splitter.get_nodes_from_documents([text_node])

    assert chunks[0].text.startswith("class Foo:")
    assert "scopes" in chunks[0].metadata
    assert chunks[0].metadata["module"] == "example.foo"
    assert chunks[0].metadata["inclusive_scopes"] == [{'name': "Foo", 'type': "class"}]
    assert chunks[0].relationships[NodeRelationship.PARENT] is None
    assert chunks[0].relationships[NodeRelationship.PREVIOUS] is None
    assert chunks[0].relationships[NodeRelationship.NEXT] == chunks[1].id
    assert chunks[0].relationships[NodeRelationship.CHILD] == [chunks[1].id, chunks[2].id]
    assert chunks[0].relationships[NodeRelationship.SOURCE] == text_node.id

    assert chunks[1].text.startswith("def foo():")
    assert "scopes" in chunks[1].metadata
    assert chunks[1].metadata["module"] == "example.foo"
    assert chunks[1].metadata["inclusive_scopes"] == [{'name': "Foo", 'type': "class"}, {'name': "foo", 'type': "function"}]
    assert chunks[1].relationships[NodeRelationship.PARENT] == chunks[0].id
    assert chunks[1].relationships[NodeRelationship.PREVIOUS] == chunks[0].id
    assert chunks[1].relationships[NodeRelationship.NEXT] == chunks[2].id
    assert chunks[1].relationships[NodeRelationship.CHILD] == []
    assert chunks[1].relationships[NodeRelationship.SOURCE] == text_node.id

    assert chunks[2].text.startswith("def baz():")
    assert "scopes" in chunks[2].metadata
    assert chunks[2].metadata["module"] == "example.foo"
    assert chunks[2].metadata["inclusive_scopes"] == [{'name': "Foo", 'type': "class"}, {'name': "baz", 'type': "function"}]
    assert chunks[2].relationships[NodeRelationship.PARENT] == chunks[0].id
    assert chunks[2].relationships[NodeRelationship.PREVIOUS] == chunks[1].id
    assert chunks[2].relationships[NodeRelationship.NEXT] is None
    assert chunks[2].relationships[NodeRelationship.CHILD] == []
    assert chunks[2].relationships[NodeRelationship.SOURCE] == text_node.id

def test_html_code_splitter() -> None:
    """Test case for code splitting using HTML"""

    if "CI" in os.environ:
        return

    code_splitter = CodeBlockNodeParser(
        language="html",
        split_on_types=["html", "head", "body", "ul"],
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

    # Test the first chunk (DOCTYPE)
    assert chunks[0].text.startswith("<!DOCTYPE html>")
    assert "scopes" in chunks[0].metadata
    assert chunks[0].metadata["scopes"] == []
    assert chunks[0].relationships[NodeRelationship.PARENT] is None
    assert chunks[0].relationships[NodeRelationship.PREVIOUS] is None
    assert chunks[0].relationships[NodeRelationship.NEXT] == chunks[1].id
    assert chunks[0].relationships[NodeRelationship.CHILD] == []

    # Test the second chunk (<html> tag and its content)
    assert chunks[1].text.startswith("<html>")
    assert chunks[1].metadata["scopes"] == []
    assert chunks[1].relationships[NodeRelationship.PARENT] == chunks[0].id  # Parent should be DOCTYPE
    assert chunks[1].relationships[NodeRelationship.PREVIOUS] == chunks[0].id
    assert chunks[1].relationships[NodeRelationship.NEXT] is None  # Assuming no more chunks after this
    assert chunks[1].relationships[NodeRelationship.CHILD] == []  # Add children if needed

    # Head chunk
    assert chunks[2].text.startswith("<head>")
    assert chunks[2].metadata["scopes"] == ["html"]
    assert chunks[2].relationships[NodeRelationship.PARENT] == chunks[1].id  # Parent should be <html>
    assert chunks[2].relationships[NodeRelationship.PREVIOUS] == chunks[1].id
    assert chunks[2].relationships[NodeRelationship.NEXT] == chunks[3].id
    assert chunks[2].relationships[NodeRelationship.CHILD] == []  # Assuming no children chunks

    # Test the fourth chunk (<body> tag and its content)
    assert chunks[3].text.startswith("<body>")
    assert chunks[3].metadata["scopes"] == ["html"]
    assert chunks[3].relationships[NodeRelationship.PARENT] == chunks[1].id  # Parent should be <html>
    assert chunks[3].relationships[NodeRelationship.PREVIOUS] == chunks[1].id
    assert chunks[3].relationships[NodeRelationship.NEXT] is None  # Assuming no more chunks after this
    assert chunks[3].relationships[NodeRelationship.CHILD] == [chunks[3].id]  # Child should be <ul>

    # Test the fifth chunk (<ul> tag and its content)
    assert chunks[4].text.startswith("<ul>")
    assert chunks[4].metadata["scopes"] == ["html", "body"]
    assert chunks[4].relationships[NodeRelationship.PARENT] == chunks[2].id  # Parent should be <body>
    assert chunks[4].relationships[NodeRelationship.PREVIOUS] == chunks[2].id
    assert chunks[4].relationships[NodeRelationship.NEXT] is None  # Assuming no more chunks after this
    assert chunks[4].relationships[NodeRelationship.CHILD] == []  # Assuming no children chunks


def test_typescript_code_splitter() -> None:
    """Test case for code splitting using TypeScript"""

    if "CI" in os.environ:
        return

    code_splitter = CodeBlockNodeParser(
        language="typescript",
        split_on_types=["FunctionDeclaration", "ClassDeclaration"],
    )

    text = """\
function foo() {
    console.log("bar");
}

function baz() {
    console.log("baz");
}"""

    chunks: List[RelatedNodeInfo] = code_splitter.split_text(text)

    # Test the first chunk (function foo)
    assert chunks[0].text.startswith("function foo()")
    assert "scopes" in chunks[0].metadata
    assert chunks[0].metadata["scopes"] == []
    assert chunks[0].relationships[NodeRelationship.PARENT] is None
    assert chunks[0].relationships[NodeRelationship.PREVIOUS] is None
    assert chunks[0].relationships[NodeRelationship.NEXT] == chunks[1].id
    assert chunks[0].relationships[NodeRelationship.CHILD] == []

    # Test the second chunk (function baz)
    assert chunks[1].text.startswith("function baz()")
    assert chunks[1].metadata["scopes"] == []
    assert chunks[1].relationships[NodeRelationship.PARENT] is None
    assert chunks[1].relationships[NodeRelationship.PREVIOUS] == chunks[0].id
    assert chunks[1].relationships[NodeRelationship.NEXT] == None
    assert chunks[1].relationships[NodeRelationship.CHILD] == []


def test_typescript_code_splitter() -> None:
    """Test case for code splitting using TypeScript"""

    if "CI" in os.environ:
        return

    code_splitter = NodeParser(
        language="typescript",
        split_on_types=["FunctionDeclaration", "ClassDeclaration"],
    )

    text = """\
function foo() {
    console.log("bar");
}

class Example {
    exampleMethod() {
        console.log("line1");
    }
}

function baz() {
    console.log("bbq");
}"""

    chunks: List[RelatedNodeInfo] = code_splitter.split_text(text)

    # Test the first chunk (function foo)
    assert chunks[0].text.startswith("function foo()")
    assert "scopes" in chunks[0].metadata
    assert chunks[0].metadata["scopes"] == []
    assert chunks[0].relationships[NodeRelationship.PARENT] is None
    assert chunks[0].relationships[NodeRelationship.PREVIOUS] is None
    assert chunks[0].relationships[NodeRelationship.NEXT] == chunks[1].id
    assert chunks[0].relationships[NodeRelationship.CHILD] == []

    # Test the second chunk (class Example)
    assert chunks[1].text.startswith("class Example")
    assert chunks[1].metadata["scopes"] == []
    assert chunks[1].relationships[NodeRelationship.PARENT] is None
    assert chunks[1].relationships[NodeRelationship.PREVIOUS] == chunks[0].id
    assert chunks[1].relationships[NodeRelationship.NEXT] == chunks[2].id
    assert chunks[1].relationships[NodeRelationship.CHILD] == [chunks[2].id]

    # Test the third chunk (exampleMethod in class Example)
    assert chunks[2].text.startswith("exampleMethod()")
    assert chunks[2].metadata["scopes"] == [{'name': "Example", 'type': "class"}]
    assert chunks[2].relationships[NodeRelationship.PARENT] == chunks[1].id
    assert chunks[2].relationships[NodeRelationship.PREVIOUS] == chunks[1].id
    assert chunks[2].relationships[NodeRelationship.NEXT] is chunks[3].id
    assert chunks[2].relationships[NodeRelationship.CHILD] == []

    # Test the fourth chunk (function baz)
    assert chunks[3].text.startswith("function baz()")
    assert chunks[3].metadata["scopes"] == []
    assert chunks[3].relationships[NodeRelationship.PARENT] is None
    assert chunks[3].relationships[NodeRelationship.PREVIOUS] == chunks[2].id
    assert chunks[3].relationships[NodeRelationship.NEXT] is None
    assert chunks[3].relationships[NodeRelationship.CHILD] == []

def test_tsx_code_splitter() -> None:
    """Test case for code splitting using TypeScript JSX (TSX)"""

    if "CI" in os.environ:
        return

    code_splitter = CodeBlockNodeParser(
        language="typescript",
        split_on_types=["FunctionDeclaration", "ClassDeclaration", "div"],
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

    chunks: List[RelatedNodeInfo] = code_splitter.split_text(text)

    # Add your debugging breakpoint here if needed
    breakpoint()

    # Test the first chunk (import statement)
    assert chunks[0].text.startswith("import React from 'react';")
    assert "scopes" in chunks[0].metadata
    assert chunks[0].metadata["scopes"] == []
    assert chunks[0].relationships[NodeRelationship.PARENT] is None
    assert chunks[0].relationships[NodeRelationship.PREVIOUS] is None
    assert chunks[0].relationships[NodeRelationship.NEXT] == chunks[1].id
    assert chunks[0].relationships[NodeRelationship.CHILD] == []

    # Test the second chunk (interface definition)
    assert chunks[1].text.startswith("interface Person")
    assert chunks[1].metadata["scopes"] == []
    assert chunks[1].relationships[NodeRelationship.PARENT] is None
    assert chunks[1].relationships[NodeRelationship.PREVIOUS] == chunks[0].id
    assert chunks[1].relationships[NodeRelationship.NEXT] == chunks[2].id
    assert chunks[1].relationships[NodeRelationship.CHILD] == []

    # Test the third chunk (ExampleComponent function definition)
    assert chunks[2].text.startswith("const ExampleComponent: React.FC = () => {")
    assert chunks[2].metadata["scopes"] == []
    assert chunks[2].relationships[NodeRelationship.PARENT] is None
    assert chunks[2].relationships[NodeRelationship.PREVIOUS] == chunks[1].id
    assert chunks[2].relationships[NodeRelationship.NEXT] is chunks[3].id  # Assuming no more chunks after this
    assert chunks[2].relationships[NodeRelationship.CHILD] == []

    # Test the fourth chunk (div)
    assert chunks[3].text.startswith("<div>")
    assert chunks[3].metadata["scopes"] == [{'name': "ExampleComponent", 'type': "function"}]
    assert chunks[3].relationships[NodeRelationship.PARENT] == chunks[2].id
    assert chunks[3].relationships[NodeRelationship.PREVIOUS] is None
    assert chunks[3].relationships[NodeRelationship.NEXT] is None  # Assuming no more chunks after this
    assert chunks[3].relationships[NodeRelationship.CHILD] is None  # Assuming no children chunks




def test_cpp_code_splitter() -> None:
    """Test case for code splitting using C++"""

    if "CI" in os.environ:
        return

    code_splitter = CodeBlockNodeParser(
        language="cpp"  # Removing chunk_lines, chunk_lines_overlap, and max_chars to focus on scopes
        split_on_types=["FunctionDefinition", "ClassDefinition"],
    )

    text = """\
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}"""

    chunks = code_splitter.split_text(text)

    # Test the first chunk (#include statement)
    assert chunks[0].text.startswith("#include <iostream>")
    assert "scopes" in chunks[0].metadata
    assert chunks[0].metadata["scopes"] == []
    assert chunks[0].relationships[NodeRelationship.PARENT] is None
    assert chunks[0].relationships[NodeRelationship.PREVIOUS] is None
    assert chunks[0].relationships[NodeRelationship.NEXT] == chunks[1].id
    assert chunks[0].relationships[NodeRelationship.CHILD] == []

    # Test the second chunk (main function definition + body)
    assert chunks[1].text.startswith("int main()")
    assert chunks[1].metadata["scopes"] == [{'name': "main", 'type': "function"}]
    assert chunks[1].relationships[NodeRelationship.PARENT] is None
    assert chunks[1].relationships[NodeRelationship.PREVIOUS] == chunks[0].id
    assert chunks[1].relationships[NodeRelationship.NEXT] is None
    assert chunks[1].relationships[NodeRelationship.CHILD] == []
