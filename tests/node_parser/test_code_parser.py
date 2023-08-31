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

    chunks: List[TextNode] = code_splitter.get_nodes_from_documents([text_node])

    # This is the module scope
    assert chunks[0].text.startswith("class Foo:")
    assert chunks[0].metadata["module"] == "example.foo"
    assert chunks[0].metadata["inclusive_scopes"] == []
    assert NodeRelationship.PARENT not in chunks[0].relationships
    assert [c.node_id for c in chunks[0].relationships[NodeRelationship.CHILD]] == [chunks[1].id_]
    assert isinstance(chunks[0].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert chunks[0].relationships[NodeRelationship.SOURCE].node_id == text_node.id_
    assert NodeRelationship.PREVIOUS not in chunks[0].relationships
    assert NodeRelationship.NEXT not in chunks[0].relationships

    # This is the class scope
    assert chunks[1].text.startswith("class Foo:")
    assert chunks[1].metadata["module"] == "example.foo"
    assert chunks[1].metadata["inclusive_scopes"] == [{'name': "Foo", 'type': "class_definition"}]
    assert isinstance(chunks[1].relationships[NodeRelationship.PARENT], RelatedNodeInfo)
    assert chunks[1].relationships[NodeRelationship.PARENT].node_id == chunks[0].id_
    assert [c.node_id for c in chunks[1].relationships[NodeRelationship.CHILD]] == [chunks[2].id_, chunks[3].id_]
    assert isinstance(chunks[1].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert chunks[1].relationships[NodeRelationship.SOURCE].node_id == text_node.id_
    assert NodeRelationship.PREVIOUS not in chunks[1].relationships
    assert NodeRelationship.NEXT not in chunks[1].relationships

    # This is the first method scope
    assert chunks[2].text.startswith("def foo():")
    assert chunks[2].metadata["module"] == "example.foo"
    assert chunks[2].metadata["inclusive_scopes"] == [{'name': "Foo", 'type': "class_definition"}, {'name': "foo", 'type': "function_definition"}]
    assert isinstance(chunks[2].relationships[NodeRelationship.PARENT], RelatedNodeInfo)
    assert chunks[2].relationships[NodeRelationship.PARENT].node_id == chunks[1].id_
    assert chunks[2].relationships[NodeRelationship.CHILD] == []
    assert isinstance(chunks[2].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert chunks[2].relationships[NodeRelationship.SOURCE].node_id == text_node.id_
    assert NodeRelationship.PREVIOUS not in chunks[2].relationships
    assert NodeRelationship.NEXT not in chunks[2].relationships

    # This is the second method scope
    assert chunks[3].text.startswith("def baz():")
    assert chunks[3].metadata["module"] == "example.foo"
    assert chunks[3].metadata["inclusive_scopes"] == [{'name': "Foo", 'type': "class_definition"}, {'name': "baz", 'type': "function_definition"}]
    assert isinstance(chunks[3].relationships[NodeRelationship.PARENT], RelatedNodeInfo)
    assert chunks[3].relationships[NodeRelationship.PARENT].node_id == chunks[1].id_
    assert chunks[3].relationships[NodeRelationship.CHILD] == []
    assert isinstance(chunks[3].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert chunks[3].relationships[NodeRelationship.SOURCE].node_id == text_node.id_
    assert NodeRelationship.PREVIOUS not in chunks[3].relationships
    assert NodeRelationship.NEXT not in chunks[3].relationships


def test_html_code_splitter() -> None:
    """Test case for code splitting using HTML"""

    if "CI" in os.environ:
        return

    code_splitter = CodeBlockNodeParser(
        language="html",
        min_characters=len("<title>My Example Page</title>")+1
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

    text_node = TextNode(
        text=text,
    )
    chunks = code_splitter.get_nodes_from_documents([text_node])

    # This is the DOCTYPE scope
    assert chunks[0].text.startswith("<!DOCTYPE html>")
    assert chunks[0].metadata["inclusive_scopes"] == []
    assert NodeRelationship.PARENT not in chunks[0].relationships
    assert [c.node_id for c in chunks[0].relationships[NodeRelationship.CHILD]] == [chunks[1].id_]
    assert chunks[0].relationships[NodeRelationship.SOURCE].node_id == text_node.id_
    assert NodeRelationship.PREVIOUS not in chunks[0].relationships
    assert NodeRelationship.NEXT not in chunks[0].relationships

    # This is the html scope
    assert chunks[1].text.startswith("<html>")
    assert chunks[1].metadata["inclusive_scopes"] == [{'name': "<html>", 'type': "element"}]
    assert chunks[1].relationships[NodeRelationship.PARENT].node_id == chunks[0].id_
    assert [c.node_id for c in chunks[1].relationships[NodeRelationship.CHILD]] == [chunks[2].id_, chunks[3].id_]
    assert chunks[1].relationships[NodeRelationship.SOURCE].node_id == text_node.id_
    assert NodeRelationship.PREVIOUS not in chunks[1].relationships
    assert NodeRelationship.NEXT not in chunks[1].relationships

    # Head chunk
    assert chunks[2].text.startswith("<head>")
    assert chunks[2].metadata["inclusive_scopes"] == [{"name": "<html>", "type": "element"}, {"name": "<head>", "type": "element"}]
    assert chunks[2].relationships[NodeRelationship.PARENT].node_id == chunks[1].id_  # Parent should be <html>
    assert [c.node_id for c in chunks[2].relationships[NodeRelationship.CHILD]] == []  # Child should be <title>
    assert chunks[2].relationships[NodeRelationship.SOURCE].node_id == text_node.id_
    assert NodeRelationship.PREVIOUS not in chunks[2].relationships
    assert NodeRelationship.NEXT not in chunks[2].relationships

    # Test the fourth chunk (<body> tag and its content)
    assert chunks[3].text.startswith("<body>")
    assert chunks[3].metadata["inclusive_scopes"] == [{"name": "<html>", "type": "element"}, {"name": "<body>", "type": "element"}]
    assert chunks[3].relationships[NodeRelationship.PARENT].node_id == chunks[1].id_  # Parent should be <html>
    assert chunks[5].id_ in [c.node_id for c in chunks[3].relationships[NodeRelationship.CHILD]]
    assert chunks[3].relationships[NodeRelationship.SOURCE].node_id == text_node.id_
    assert NodeRelationship.PREVIOUS not in chunks[3].relationships
    assert NodeRelationship.NEXT not in chunks[3].relationships

    # Test the seventh chunk (<ul> tag and its content)
    assert chunks[6].text.startswith("<ul>")
    assert chunks[6].metadata["inclusive_scopes"] == [{"name": "<html>", "type": "element"}, {"name": "<body>", "type": "element"}, {"name": "<ul>", "type": "element"}]
    assert chunks[6].relationships[NodeRelationship.PARENT].node_id == chunks[3].id_  # Parent should be <body>
    assert [c.node_id for c in chunks[6].relationships[NodeRelationship.CHILD]] == []
    assert chunks[6].relationships[NodeRelationship.SOURCE].node_id == text_node.id_
    assert NodeRelationship.PREVIOUS not in chunks[6].relationships
    assert NodeRelationship.NEXT not in chunks[6].relationships


def test_typescript_code_splitter() -> None:
    """Test case for code splitting using TypeScript"""

    if "CI" in os.environ:
        return

    code_splitter = CodeBlockNodeParser(
        language="typescript",
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

    text_node = TextNode(
        text=text,
    )
    chunks: List[RelatedNodeInfo] = code_splitter.get_nodes_from_documents([text_node])

    # Test the second chunk (function foo)
    assert chunks[1].text.startswith("function foo()")
    assert chunks[1].metadata["inclusive_scopes"] == [{"name": "foo", "type": "function_declaration"}]
    assert chunks[1].relationships[NodeRelationship.PARENT].node_id == chunks[0].id_
    assert [c.node_id for c in chunks[1].relationships[NodeRelationship.CHILD]] == []

    # Test the third chunk (class Example)
    assert chunks[2].text.startswith("class Example")
    assert chunks[2].metadata["inclusive_scopes"] == [{"name": "Example", "type": "class_declaration"}]
    assert chunks[2].relationships[NodeRelationship.PARENT].node_id == chunks[0].id_
    assert [c.node_id for c in chunks[2].relationships[NodeRelationship.CHILD]] == [chunks[3].id_]

    # Test the fourth chunk (exampleMethod in class Example)
    assert chunks[3].text.startswith("exampleMethod()")
    assert chunks[3].metadata["inclusive_scopes"] == [{'name': "Example", 'type': "class_declaration"}, {'name': "exampleMethod", 'type': "method_definition"}]
    assert chunks[3].relationships[NodeRelationship.PARENT].node_id == chunks[2].id_
    assert chunks[3].relationships[NodeRelationship.CHILD] == []

    # Test the fifth chunk (function baz)
    assert chunks[4].text.startswith("function baz()")
    assert chunks[4].metadata["inclusive_scopes"] == [{'name': "baz", 'type': "function_declaration"}]
    assert chunks[4].relationships[NodeRelationship.PARENT].node_id == chunks[0].id_
    assert chunks[4].relationships[NodeRelationship.CHILD] == []

def test_tsx_code_splitter() -> None:
    """Test case for code splitting using TypeScript JSX (TSX)"""

    if "CI" in os.environ:
        return

    code_splitter = CodeBlockNodeParser(
        language="typescript",
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

    text_node = TextNode(
        text=text,
    )
    chunks: List[RelatedNodeInfo] = code_splitter.get_nodes_from_documents([text_node])

    # Test the first chunk (import statement)
    assert chunks[0].text.startswith("import React from 'react';")
    assert chunks[0].metadata["inclusive_scopes"] == []

    # Test the second chunk (interface definition)
    assert chunks[1].text.startswith("interface Person")
    assert chunks[1].metadata["inclusive_scopes"] == [{'name': "Person", 'type': "interface_declaration"}]
    assert chunks[1].relationships[NodeRelationship.PARENT].node_id == chunks[0].id_
    assert chunks[1].relationships[NodeRelationship.CHILD] == []

    # Test the third chunk (ExampleComponent function definition)
    assert chunks[2].text.startswith("const ExampleComponent: React.FC = () => {")
    assert chunks[2].metadata["inclusive_scopes"] == [{'name': "ExampleComponent", 'type': "lexical_declaration"}]
    assert chunks[2].relationships[NodeRelationship.PARENT].node_id == chunks[0].id_

    # TODO: Unfortunately tree_splitter errors on the html elements

def test_cpp_code_splitter() -> None:
    """Test case for code splitting using C++"""

    if "CI" in os.environ:
        return

    code_splitter = CodeBlockNodeParser(
        language="cpp",  # Removing chunk_lines, chunk_lines_overlap, and max_chars to focus on scopes
    )

    text = """\
#include <iostream>

class MyClass {       // The class
  public:             // Access specifier
    int myNum;        // Attribute (int variable)
    string myString;  // Attribute (string variable)
    void myMethod() { // Method/function defined inside the class
        cout << "Hello World!";
    }
};

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}"""

    text_node = TextNode(
        text=text,
    )
    chunks = code_splitter.get_nodes_from_documents([text_node])

    # Test the first chunk (#include statement)
    assert chunks[0].text.startswith("#include <iostream>")
    assert chunks[0].metadata["inclusive_scopes"] == []

    # Test the second chunk (class MyClass)
    assert chunks[1].text.startswith("class MyClass")
    assert chunks[1].metadata["inclusive_scopes"] == [{'name': "MyClass", 'type': "class_specifier"}]
    assert chunks[1].relationships[NodeRelationship.PARENT].node_id == chunks[0].id_
    assert [c.node_id for c in chunks[1].relationships[NodeRelationship.CHILD]] == [chunks[2].id_]

    # Test the third chunk (myMethod in class MyClass)
    assert chunks[2].text.startswith("void myMethod()")
    assert chunks[2].metadata["inclusive_scopes"] == [{'name': "MyClass", 'type': "class_specifier"}, {'name': "myMethod()", 'type': "function_definition"}]
    assert chunks[2].relationships[NodeRelationship.PARENT].node_id == chunks[1].id_
    assert chunks[2].relationships[NodeRelationship.CHILD] == []

    # Test the fourth chunk (main function)
    assert chunks[3].text.startswith("int main()")
    assert chunks[3].metadata["inclusive_scopes"] == [{'name': "main()", 'type': "function_definition"}]
    assert chunks[3].relationships[NodeRelationship.PARENT].node_id == chunks[0].id_
    assert chunks[3].relationships[NodeRelationship.CHILD] == []
