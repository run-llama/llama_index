"""Test CodeHierarchyNodeParser with skeleton option set to False."""

import os
from typing import List, cast

from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode


def test_python_code_splitter() -> None:
    """Test case for code splitting using python."""
    if "CI" in os.environ:
        return

    code_splitter = CodeHierarchyNodeParser(
        language="python", skeleton=True, chunk_min_characters=0
    )

    text = """\
class Foo:
    def bar() -> None:
        print("bar")

    async def baz():
        print("baz")"""

    text_node = TextNode(
        text=text,
        metadata={
            "module": "example.foo",
        },
    )

    chunks: List[TextNode] = code_splitter.get_nodes_from_documents([text_node])

    # This is the module scope
    assert (
        chunks[0].text
        == f"""\
class Foo:
    # {CodeHierarchyNodeParser._get_comment_text(chunks[1])}"""
    )
    assert chunks[0].metadata["module"] == "example.foo"
    assert chunks[0].metadata["inclusive_scopes"] == []
    assert NodeRelationship.PARENT not in chunks[0].relationships
    assert [c.node_id for c in chunks[0].relationships[NodeRelationship.CHILD]] == [
        chunks[1].id_
    ]
    assert isinstance(chunks[0].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[0].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[0].relationships
    assert NodeRelationship.NEXT not in chunks[0].relationships

    # This is the class scope
    assert (
        chunks[1].text
        == f"""\
class Foo:
    def bar() -> None:
        # {CodeHierarchyNodeParser._get_comment_text(chunks[2])}

    async def baz():
        # {CodeHierarchyNodeParser._get_comment_text(chunks[3])}"""
    )
    assert chunks[1].metadata["module"] == "example.foo"
    assert chunks[1].metadata["inclusive_scopes"] == [
        {"name": "Foo", "type": "class_definition", "signature": "class Foo:"}
    ]
    assert isinstance(chunks[1].relationships[NodeRelationship.PARENT], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[1].relationships[NodeRelationship.PARENT]).node_id
        == chunks[0].id_
    )
    assert [c.node_id for c in chunks[1].relationships[NodeRelationship.CHILD]] == [
        chunks[2].id_,
        chunks[3].id_,
    ]
    assert isinstance(chunks[1].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[1].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[1].relationships
    assert NodeRelationship.NEXT not in chunks[1].relationships

    # This is the first method scope
    assert (
        chunks[2].text
        == """\
    def bar() -> None:
        print("bar")"""
    )
    assert chunks[2].metadata["module"] == "example.foo"
    assert chunks[2].metadata["inclusive_scopes"] == [
        {"name": "Foo", "type": "class_definition", "signature": "class Foo:"},
        {
            "name": "bar",
            "type": "function_definition",
            "signature": "def bar() -> None:",
        },
    ]
    assert isinstance(chunks[2].relationships[NodeRelationship.PARENT], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[2].relationships[NodeRelationship.PARENT]).node_id
        == chunks[1].id_
    )
    assert chunks[2].relationships[NodeRelationship.CHILD] == []
    assert isinstance(chunks[2].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[2].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[2].relationships
    assert NodeRelationship.NEXT not in chunks[2].relationships

    # This is the second method scope
    assert (
        chunks[3].text
        == """\
    async def baz():
        print("baz")"""
    )
    assert chunks[3].metadata["module"] == "example.foo"
    assert chunks[3].metadata["inclusive_scopes"] == [
        {"name": "Foo", "type": "class_definition", "signature": "class Foo:"},
        {"name": "baz", "type": "function_definition", "signature": "async def baz():"},
    ]
    assert isinstance(chunks[3].relationships[NodeRelationship.PARENT], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[3].relationships[NodeRelationship.PARENT]).node_id
        == chunks[1].id_
    )
    assert chunks[3].relationships[NodeRelationship.CHILD] == []
    assert isinstance(chunks[3].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[3].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[3].relationships
    assert NodeRelationship.NEXT not in chunks[3].relationships


def test_python_code_splitter_with_decorators() -> None:
    """Test case for code splitting using python."""
    if "CI" in os.environ:
        return

    code_splitter = CodeHierarchyNodeParser(
        language="python", skeleton=True, chunk_min_characters=0
    )

    text = """\
@foo
class Foo:
    @bar
    @barfoo
    def bar() -> None:
        print("bar")"""

    text_node = TextNode(
        text=text,
        metadata={
            "module": "example.foo",
        },
    )

    chunks: List[TextNode] = code_splitter.get_nodes_from_documents([text_node])

    # This is the module scope
    assert (
        chunks[0].text
        == f"""\
@foo
class Foo:
    # {CodeHierarchyNodeParser._get_comment_text(chunks[1])}"""
    )
    assert chunks[0].metadata["module"] == "example.foo"
    assert chunks[0].metadata["inclusive_scopes"] == []
    assert NodeRelationship.PARENT not in chunks[0].relationships
    assert [c.node_id for c in chunks[0].relationships[NodeRelationship.CHILD]] == [
        chunks[1].id_
    ]
    assert isinstance(chunks[0].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[0].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[0].relationships
    assert NodeRelationship.NEXT not in chunks[0].relationships

    # This is the class scope
    assert (
        chunks[1].text
        == f"""\
class Foo:
    @bar
    @barfoo
    def bar() -> None:
        # {CodeHierarchyNodeParser._get_comment_text(chunks[2])}"""
    )
    assert chunks[1].metadata["module"] == "example.foo"
    assert chunks[1].metadata["inclusive_scopes"] == [
        {"name": "Foo", "type": "class_definition", "signature": "class Foo:"}
    ]
    assert isinstance(chunks[1].relationships[NodeRelationship.PARENT], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[1].relationships[NodeRelationship.PARENT]).node_id
        == chunks[0].id_
    )
    assert [c.node_id for c in chunks[1].relationships[NodeRelationship.CHILD]] == [
        chunks[2].id_,
    ]
    assert isinstance(chunks[1].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[1].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[1].relationships
    assert NodeRelationship.NEXT not in chunks[1].relationships

    # This is the first method scope
    assert (
        chunks[2].text
        == """\
    def bar() -> None:
        print("bar")"""
    )
    assert chunks[2].metadata["module"] == "example.foo"
    assert chunks[2].metadata["inclusive_scopes"] == [
        {"name": "Foo", "type": "class_definition", "signature": "class Foo:"},
        {
            "name": "bar",
            "type": "function_definition",
            "signature": "def bar() -> None:",
        },
    ]
    assert isinstance(chunks[2].relationships[NodeRelationship.PARENT], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[2].relationships[NodeRelationship.PARENT]).node_id
        == chunks[1].id_
    )
    assert chunks[2].relationships[NodeRelationship.CHILD] == []
    assert isinstance(chunks[2].relationships[NodeRelationship.SOURCE], RelatedNodeInfo)
    assert (
        cast(RelatedNodeInfo, chunks[2].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[2].relationships
    assert NodeRelationship.NEXT not in chunks[2].relationships


def test_html_code_splitter() -> None:
    """Test case for code splitting using HTML."""
    if "CI" in os.environ:
        return

    code_splitter = CodeHierarchyNodeParser(
        language="html",
        chunk_min_characters=len("    <title>My Example Page</title>") + 1,
        skeleton=True,
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
    assert (
        chunks[0].text
        == f"""\
<!DOCTYPE html>
<html>
    <!-- {CodeHierarchyNodeParser._get_comment_text(chunks[1])} -->
</html>"""
    )
    assert chunks[0].metadata["inclusive_scopes"] == []
    assert NodeRelationship.PARENT not in chunks[0].relationships
    assert [c.node_id for c in chunks[0].relationships[NodeRelationship.CHILD]] == [
        chunks[1].id_
    ]
    assert (
        cast(RelatedNodeInfo, chunks[0].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[0].relationships
    assert NodeRelationship.NEXT not in chunks[0].relationships

    # This is the html scope
    assert (
        chunks[1].text
        == f"""\
<html>
<head>
    <!-- {CodeHierarchyNodeParser._get_comment_text(chunks[2])} -->
</head>
<body>
    <!-- {CodeHierarchyNodeParser._get_comment_text(chunks[3])} -->
</body>
</html>"""
    )
    assert chunks[1].metadata["inclusive_scopes"] == [
        {"name": "html", "type": "element", "signature": "<html>"}
    ]
    assert (
        cast(RelatedNodeInfo, chunks[1].relationships[NodeRelationship.PARENT]).node_id
        == chunks[0].id_
    )
    assert [c.node_id for c in chunks[1].relationships[NodeRelationship.CHILD]] == [
        chunks[2].id_,
        chunks[3].id_,
    ]
    assert (
        cast(RelatedNodeInfo, chunks[1].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[1].relationships
    assert NodeRelationship.NEXT not in chunks[1].relationships

    # Head chunk
    assert (
        chunks[2].text
        == """\
<head>
    <title>My Example Page</title>
</head>"""
    )
    assert chunks[2].metadata["inclusive_scopes"] == [
        {"name": "html", "type": "element", "signature": "<html>"},
        {"name": "head", "type": "element", "signature": "<head>"},
    ]
    assert (
        cast(RelatedNodeInfo, chunks[2].relationships[NodeRelationship.PARENT]).node_id
        == chunks[1].id_
    )  # Parent should be <html>
    assert [
        c.node_id for c in chunks[2].relationships[NodeRelationship.CHILD]
    ] == []  # Child should be <title>
    assert (
        cast(RelatedNodeInfo, chunks[2].relationships[NodeRelationship.SOURCE]).node_id
        == text_node.id_
    )
    assert NodeRelationship.PREVIOUS not in chunks[2].relationships
    assert NodeRelationship.NEXT not in chunks[2].relationships


def test_typescript_code_splitter() -> None:
    """Test case for code splitting using TypeScript."""
    if "CI" in os.environ:
        return

    code_splitter = CodeHierarchyNodeParser(
        language="typescript", skeleton=True, chunk_min_characters=0
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

    # Fstrings don't like forward slash
    double_forward_slash: str = "//"
    assert (
        chunks[0].text
        == f"""\
function foo() {{
    {double_forward_slash} {CodeHierarchyNodeParser._get_comment_text(chunks[1])}
}}

class Example {{
    {double_forward_slash} {CodeHierarchyNodeParser._get_comment_text(chunks[2])}
}}

function baz() {{
    {double_forward_slash} {CodeHierarchyNodeParser._get_comment_text(chunks[4])}
}}"""
    )

    # Test the second chunk (function foo)
    assert (
        chunks[1].text
        == """\
function foo() {
    console.log("bar");
}"""
    )
    assert chunks[1].metadata["inclusive_scopes"] == [
        {"name": "foo", "type": "function_declaration", "signature": "function foo()"}
    ]
    assert (
        cast(RelatedNodeInfo, chunks[1].relationships[NodeRelationship.PARENT]).node_id
        == chunks[0].id_
    )
    assert [c.node_id for c in chunks[1].relationships[NodeRelationship.CHILD]] == []

    # Test the third chunk (class Example)
    assert (
        chunks[2].text
        == f"""\
class Example {{
    exampleMethod() {{
        {double_forward_slash} {CodeHierarchyNodeParser._get_comment_text(chunks[3])}
    }}
}}"""
    )
    assert chunks[2].metadata["inclusive_scopes"] == [
        {"name": "Example", "type": "class_declaration", "signature": "class Example"}
    ]
    assert (
        cast(RelatedNodeInfo, chunks[2].relationships[NodeRelationship.PARENT]).node_id
        == chunks[0].id_
    )
    assert [c.node_id for c in chunks[2].relationships[NodeRelationship.CHILD]] == [
        chunks[3].id_
    ]

    # Test the fourth chunk (exampleMethod in class Example)
    assert (
        chunks[3].text
        == """\
    exampleMethod() {
        console.log("line1");
    }"""
    )
    assert chunks[3].metadata["inclusive_scopes"] == [
        {"name": "Example", "type": "class_declaration", "signature": "class Example"},
        {
            "name": "exampleMethod",
            "type": "method_definition",
            "signature": "exampleMethod()",
        },
    ]
    assert (
        cast(RelatedNodeInfo, chunks[3].relationships[NodeRelationship.PARENT]).node_id
        == chunks[2].id_
    )
    assert chunks[3].relationships[NodeRelationship.CHILD] == []

    # Test the fifth chunk (function baz)
    assert (
        chunks[4].text
        == """\
function baz() {
    console.log("bbq");
}"""
    )
    assert chunks[4].metadata["inclusive_scopes"] == [
        {"name": "baz", "type": "function_declaration", "signature": "function baz()"}
    ]
    assert (
        cast(RelatedNodeInfo, chunks[4].relationships[NodeRelationship.PARENT]).node_id
        == chunks[0].id_
    )
    assert chunks[4].relationships[NodeRelationship.CHILD] == []


# No need to test everything that is in test_code_hierarchy_no_skeleton


def test_typescript_code_splitter_2() -> None:
    """Test case for code splitting using TypeScript."""
    if "CI" in os.environ:
        return

    code_splitter = CodeHierarchyNodeParser(
        language="typescript", skeleton=True, chunk_min_characters=0
    )

    text = """\
class Example {
    exampleMethod() {
        console.log("line1");
    }
}
"""

    text_node = TextNode(
        text=text,
    )
    chunks: List[RelatedNodeInfo] = code_splitter.get_nodes_from_documents([text_node])

    # Fstrings don't like forward slash
    double_forward_slash: str = "//"
    assert (
        chunks[0].text
        == f"""\
class Example {{
    {double_forward_slash} {CodeHierarchyNodeParser._get_comment_text(chunks[1])}
}}
"""
    )


def test_skeletonize_with_repeated_function() -> None:
    """Test case for code splitting using python."""
    if "CI" in os.environ:
        return

    code_splitter = CodeHierarchyNodeParser(
        language="python", skeleton=True, chunk_min_characters=0
    )

    text = """\
def _handle_extra_radiation_types(datetime_or_doy, epoch_year):
    if np.isscalar(datetime_or_doy):
        def to_doy(x): return x                                 # noqa: E306
        to_datetimeindex = partial(tools._doy_to_datetimeindex,
                                   epoch_year=epoch_year)
        to_output = tools._scalar_out
    else:
        def to_doy(x): return x                                 # noqa: E306
        to_datetimeindex = partial(tools._doy_to_datetimeindex,
                                   epoch_year=epoch_year)
        to_output = tools._array_out

    return to_doy, to_datetimeindex, to_output"""

    text_node = TextNode(
        text=text,
        metadata={
            "module": "example.foo",
        },
    )

    chunks: List[TextNode] = code_splitter.get_nodes_from_documents([text_node])
    assert len(chunks) == 4
    assert (
        chunks[0].text
        == f"""def _handle_extra_radiation_types(datetime_or_doy, epoch_year):
    # {CodeHierarchyNodeParser._get_comment_text(chunks[1])}"""
    )

    assert (
        chunks[1].text
        == f"""def _handle_extra_radiation_types(datetime_or_doy, epoch_year):
    if np.isscalar(datetime_or_doy):
        def to_doy(x):
                # {CodeHierarchyNodeParser._get_comment_text(chunks[2])}
        to_datetimeindex = partial(tools._doy_to_datetimeindex,
                                   epoch_year=epoch_year)
        to_output = tools._scalar_out
    else:
        def to_doy(x):
                # {CodeHierarchyNodeParser._get_comment_text(chunks[3])}
        to_datetimeindex = partial(tools._doy_to_datetimeindex,
                                   epoch_year=epoch_year)
        to_output = tools._array_out

    return to_doy, to_datetimeindex, to_output"""
    )
    assert (
        chunks[2].text
        == """        def to_doy(x): return x                                 # noqa: E306"""
    )
    assert (
        chunks[3].text
        == """        def to_doy(x): return x                                 # noqa: E306"""
    )
