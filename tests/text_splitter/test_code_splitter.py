"""Test text splitter."""
import os

from llama_index.text_splitter import CodeSplitter


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
    assert chunks[0] == "def foo():\n    print(\"bar\")"
    assert chunks[1] == "def baz():\n    print(\"bbq\")"


# def test_typescript_code_splitter() -> None:
#     """Test case for code splitting using typescript"""

#     if "CI" in os.environ:
#         return

#     code_splitter = CodeSplitter(
#         language="typescript", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
#     )

#     text = """\
# function foo() {
#     console.log("bar");
# }

# function baz() {
#     console.log("bbq");
# }"""

#     chunks = code_splitter.split_text(text)
#     assert chunks[0].startswith("function foo()")
#     assert chunks[1].startswith("function baz()")


# def test_html_code_splitter() -> None:
#     """Test case for code splitting using typescript"""

#     if "CI" in os.environ:
#         return

#     code_splitter = CodeSplitter(
#         language="html", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
#     )

#     text = """\
# <!DOCTYPE html>
# <html>
# <head>
#     <title>My Example Page</title>
# </head>
# <body>
#     <h1>Welcome to My Example Page</h1>
#     <p>This is a basic HTML page example.</p>
#     <ul>
#         <li>Item 1</li>
#         <li>Item 2</li>
#         <li>Item 3</li>
#     </ul>
#     <img src="https://example.com/image.jpg" alt="Example Image">
# </body>
# </html>"""

#     chunks = code_splitter.split_text(text)
#     assert chunks[0].startswith("<!DOCTYPE html>")
#     assert chunks[1].startswith("<html>")
#     assert chunks[2].startswith("<head>")


# def test_tsx_code_splitter() -> None:
#     """Test case for code splitting using typescript"""

#     if "CI" in os.environ:
#         return

#     code_splitter = CodeSplitter(
#         language="typescript", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
#     )

#     text = """\
# import React from 'react';

# interface Person {
#   name: string;
#   age: number;
# }

# const ExampleComponent: React.FC = () => {
#   const person: Person = {
#     name: 'John Doe',
#     age: 30,
#   };

#   return (
#     <div>
#       <h1>Hello, {person.name}!</h1>
#       <p>You are {person.age} years old.</p>
#     </div>
#   );
# };

# export default ExampleComponent;"""

#     chunks = code_splitter.split_text(text)
#     assert chunks[0].startswith("import React from 'react';")
#     assert chunks[1].startswith("interface Person")


# def test_cpp_code_splitter() -> None:
#     """Test case for code splitting using typescript"""

#     if "CI" in os.environ:
#         return

#     code_splitter = CodeSplitter(
#         language="cpp", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
#     )

#     text = """\
# #include <iostream>

# int main() {
#     std::cout << "Hello, World!" << std::endl;
#     return 0;
# }"""

#     chunks = code_splitter.split_text(text)
#     assert chunks[0].startswith("#include <iostream>")
#     assert chunks[1].startswith("int main()")
#     assert chunks[2].startswith("{\n    std::cout")
