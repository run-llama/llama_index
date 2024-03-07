"""Test CodeHierarchyNodeParser reading itself."""
from typing import Sequence
from llama_index.core import SimpleDirectoryReader
from pytest import fixture
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser
from llama_index.core.text_splitter import CodeSplitter
from pathlib import Path
from llama_index.core.schema import BaseNode

from IPython.display import Markdown, display

from llama_index.packs.code_hierarchy import CodeHierarchyKeywordQueryEngine
from llama_index.core.llama_pack import download_llama_pack

def print_python(python_text: str) -> None:
    """This function prints python text in ipynb nicely formatted."""
    print("```python\n" + python_text + "```")


@fixture()
def code_hierarchy_nodes():
    reader = SimpleDirectoryReader(
        input_files=[Path(__file__).parent / Path("../llama_index/packs/code_hierarchy/code_hierarchy.py")],
        file_metadata=lambda x: {"filepath": x},
    )
    nodes = reader.load_data()
    return CodeHierarchyNodeParser(
        language="python",
        # You can further parameterize the CodeSplitter to split the code
        # into "chunks" that match your context window size using
        # chunck_lines and max_chars parameters, here we just use the defaults
        code_splitter=CodeSplitter(language="python", max_chars=1000, chunk_lines=10),
    ).get_nodes_from_documents(nodes)


def test_code_splitter_NEXT_relationship_indention(
    code_hierarchy_nodes: Sequence[BaseNode],
) -> None:
    """When using jupyter I found that the final brevity comment was indented when it shouldnt be."""
    print_python(code_hierarchy_nodes[0].text)
    assert (
        not code_hierarchy_nodes[0].text.split("\n")[-1].startswith(" ")
    ), "The last line should not be indented"


def test_query_by_module_name(code_hierarchy_nodes: Sequence[BaseNode]) -> None:
    """Test querying the index by filename."""
    index = CodeHierarchyKeywordQueryEngine(nodes=code_hierarchy_nodes)
    query = "code_hierarchy"
    results = index.query(query)
    assert len(results.response) >= 1


def test_query_by_name(code_hierarchy_nodes: Sequence[BaseNode]) -> None:
    """Test querying the index by signature."""
    index = CodeHierarchyKeywordQueryEngine(nodes=code_hierarchy_nodes)
    query = "CodeHierarchyNodeParser"
    results = index.query(query)
    assert len(results.response) >= 1


def test_get_tool(code_hierarchy_nodes: Sequence[BaseNode]) -> None:
    """Test querying the index by signature."""
    index = CodeHierarchyKeywordQueryEngine(nodes=code_hierarchy_nodes)
    query = "CodeHierarchyNodeParser"
    results = index.as_langchain_tool().run(query)
    assert len(results) >= 1