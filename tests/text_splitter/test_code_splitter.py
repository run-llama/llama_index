"""Test text splitter."""
import os

from llama_index.text_splitter.code_splitter import ELLIPSES_COMMENT, CodeSplitter, _generate_comment_line


def test_generate_comment_line_python() -> None:
    assert _generate_comment_line('Python', 'This is a Python comment') == '# This is a Python comment'

def test_generate_comment_line_c_lowercase() -> None:
    assert _generate_comment_line('c', 'This is a C comment') == '// This is a C comment'

def test_generate_comment_line_java() -> None:
    assert _generate_comment_line('Java', 'This is a Java comment') == '// This is a Java comment'

def test_generate_comment_line_html() -> None:
    assert _generate_comment_line('HTML', 'This is an HTML comment') == '<!-- This is an HTML comment -->'

def test_generate_comment_line_unknown() -> None:
    assert _generate_comment_line('Unknown', 'This is an unknown language comment') == '🦙This is an unknown language comment🦙'

def test_generate_comment_line_unknown_random() -> None:
    assert _generate_comment_line('asdf', 'This is an unknown language comment') == '🦙This is an unknown language comment🦙'


def test_python_code_splitter() -> None:
    """Test case for code splitting using python"""

    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="python"
    )

    text = """\
def foo():
    print("bar")

def baz():
    print("bbq")"""

    chunks = code_splitter.split_text(text)
    assert chunks[0] == "def foo():\n    print(\"bar\")"
    assert chunks[1] == "def baz():\n    print(\"bbq\")"


def test_python_code_splitter_class() -> None:
    """Test case for code splitting using python including classes"""

    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="python"
    )

    text = """\
class Foo:
    a: int = 1
    def foo(self):
        print("foo")
    def bar(self):
        print("bar")"""

    chunks = code_splitter.split_text(text)
    ellipsis_comment = _generate_comment_line('Python', ELLIPSES_COMMENT)
    assert chunks[0] == f"class Foo:\n{ellipsis_comment}\n    def foo(self):\n        print(\"foo\")"
    assert chunks[1] == f"class Foo:\n{ellipsis_comment}\n    def bar(self):\n        print(\"bar\")"
    assert chunks[2] == f"class Foo:\n    a: int = 1\n{ellipsis_comment}"

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
