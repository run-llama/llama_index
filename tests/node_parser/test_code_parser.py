"""Test text splitter."""
import os
from typing import List

from llama_index.node_parser import CodeParser
from llama_index.schema import NodeRelationship, RelatedNodeInfo

def test_python_code_splitter() -> None:
    """Test case for code splitting using python"""

    if "CI" in os.environ:
        return

    code_splitter = CodeParser(
        language="python", chunk_lines=4, chunk_lines_overlap=1, max_chars=30
    )

    text = """\
class Foo:
    def foo():
        print("bar")

    def baz():
        print("line1")
        print("line2")
        print("line3")
        print("line4")
        print("line5")
        """

    chunks: List[RelatedNodeInfo] = code_splitter.split_text(text)
    assert chunks[0].text.startswith("class Foo:")
    assert "scopes" in chunks[0].metadata
    assert chunks[0].metadata["scopes"] == []
    assert chunks[0].relationships[NodeRelationship.PARENT] is None
    assert chunks[0].relationships[NodeRelationship.PREVIOUS] is None
    assert chunks[0].relationships[NodeRelationship.NEXT] == chunks[1].id
    assert chunks[0].[NodeRelationship.CHILDREN] == [chunks[1].id, chunks[2].id]

    assert chunks[1].text.startswith("def foo():")
    assert chunks[1].metadata["scopes"] == [{'name': "Foo", 'type': "class"}]
    assert chunks[1].relationships[NodeRelationship.PARENT] == chunks[0].id
    assert chunks[1].relationships[NodeRelationship.PREVIOUS] == chunks[0].id
    assert chunks[2].text.startswith("def baz():")
    assert chunks[2].metadata["scopes"] == [{'name': "Foo", 'type': "class"}]
    assert chunks[3].text.startswith("print(\"line2\")")
    assert chunks[3].metadata["scopes"] == [{'name': "Foo", 'type': "class"}, {'name': "baz", 'type': "functiondef"}]


def test_typescript_code_splitter() -> None:
    """Test case for code splitting using typescript"""

    if "CI" in os.environ:
        return

    code_splitter = CodeParser(
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


def test_html_code_splitter() -> None:
    """Test case for code splitting using typescript"""

    if "CI" in os.environ:
        return

    code_splitter = CodeParser(
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


def test_tsx_code_splitter() -> None:
    """Test case for code splitting using typescript"""

    if "CI" in os.environ:
        return

    code_splitter = CodeParser(
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


def test_cpp_code_splitter() -> None:
    """Test case for code splitting using typescript"""

    if "CI" in os.environ:
        return

    code_splitter = CodeParser(
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
