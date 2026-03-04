"""Test CodeHierarchyNodeParser reading itself."""

from typing import Sequence

import pytest
from llama_index.core import SimpleDirectoryReader
from pytest import fixture
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser
from llama_index.core.text_splitter import CodeSplitter
from pathlib import Path
from llama_index.core.schema import BaseNode
import re

from llama_index.packs.code_hierarchy import CodeHierarchyKeywordQueryEngine


def print_python(python_text: str) -> None:
    """This function prints python text in ipynb nicely formatted."""
    print("```python\n" + python_text + "```")


@fixture(params=[(80, 1000, 10), (500, 5000, 100)])
def code_hierarchy_nodes(request) -> Sequence[BaseNode]:
    reader = SimpleDirectoryReader(
        input_files=[
            Path(__file__).parent
            / Path("../llama_index/packs/code_hierarchy/code_hierarchy.py")
        ],
        file_metadata=lambda x: {"filepath": x},
    )
    nodes = reader.load_data()
    return CodeHierarchyNodeParser(
        language="python",
        chunk_min_characters=request.param[0],
        # You can further parameterize the CodeSplitter to split the code
        # into "chunks" that match your context window size using
        # chunck_lines and max_chars parameters, here we just use the defaults
        code_splitter=CodeSplitter(
            language="python", max_chars=request.param[1], chunk_lines=request.param[2]
        ),
    ).get_nodes_from_documents(nodes)


def test_code_splitter_NEXT_relationship_indention(
    code_hierarchy_nodes: Sequence[BaseNode],
) -> None:
    """When using jupyter I found that the final brevity comment was indented when it shouldn't be."""
    for node in code_hierarchy_nodes:
        last_line = node.text.split("\n")[-1]
        if "Code replaced for brevity" in last_line and "NEXT" in node.relationships:
            assert not last_line.startswith(" ")
            assert not last_line.startswith("\t")


def test_query_by_module_name(code_hierarchy_nodes: Sequence[BaseNode]) -> None:
    """Test querying the index by filename."""
    index = CodeHierarchyKeywordQueryEngine(nodes=code_hierarchy_nodes)
    query = "code_hierarchy"
    results = index.query(query)
    assert len(results.response) >= 1 and results.response != "None"


@pytest.mark.parametrize(
    "name",
    [
        "CodeHierarchyNodeParser",
        "_parse_node",
        "recur",
        "__init__",
    ],
)
def test_query_by_item_name(
    name: str, code_hierarchy_nodes: Sequence[BaseNode]
) -> None:
    """Test querying the index by signature."""
    index = CodeHierarchyKeywordQueryEngine(nodes=code_hierarchy_nodes)
    query = "CodeHierarchyNodeParser"
    results = index.query(query)
    assert len(results.response) >= 1 and results.response != "None"


def test_query_by_all_uuids(code_hierarchy_nodes: Sequence[BaseNode]) -> None:
    """Test querying the index by signature."""
    index = CodeHierarchyKeywordQueryEngine(nodes=code_hierarchy_nodes)
    for node in code_hierarchy_nodes:
        # Find all uuids in the node
        uuids = re.findall(r"[\w-]{36}", node.text)
        for uuid in uuids:
            results = index.query(uuid)
            assert len(results.response) >= 1 and results.response != "None"
