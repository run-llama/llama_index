# CodeHierarchyAgentPack

```bash
# install
pip install llama-index-packs-code-hierarchy

# download source code
llamaindex-cli download-llamapack CodeHierarchyAgentPack -d ./code_hierarchy_pack
```

The `CodeHierarchyAgentPack` is useful to split long code files into more reasonable chunks, while creating an agent on top to navigate the code. What this will do is create a "Hierarchy" of sorts, where sections of the code are made more reasonable by replacing the scope body with short comments telling the LLM to search for a referenced node if it wants to read that context body.

Nodes in this hierarchy will be split based on scope, like function, class, or method scope, and will have links to their children and parents so the LLM can traverse the tree.

```python
from llama_index.core.text_splitter import CodeSplitter
from llama_index.llms.openai import OpenAI
from llama_index.packs.code_hierarchy import (
    CodeHierarchyAgentPack,
    CodeHierarchyNodeParser,
)

llm = OpenAI(model="gpt-4", temperature=0.2)

documents = SimpleDirectoryReader(
    input_files=[
        Path("../llama_index/packs/code_hierarchy/code_hierarchy.py")
    ],
    file_metadata=lambda x: {"filepath": x},
).load_data()

split_nodes = CodeHierarchyNodeParser(
    language="python",
    # You can further parameterize the CodeSplitter to split the code
    # into "chunks" that match your context window size using
    # chunck_lines and max_chars parameters, here we just use the defaults
    code_splitter=CodeSplitter(
        language="python", max_chars=1000, chunk_lines=10
    ),
).get_nodes_from_documents(documents)

pack = CodeHierarchyAgentPack(split_nodes=split_nodes, llm=llm)

pack.run(
    "How does the get_code_hierarchy_from_nodes function from the code hierarchy node parser work? Provide specific implementation details."
)
```

A full example can be found [here in combination with `](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-code-hierarchy/examples/CodeHierarchyNodeParserUsage.ipynb).

## Repo Maps

The pack contains a `CodeHierarchyKeywordQueryEngine` that uses a `CodeHierarchyNodeParser` to generate a map of a repository's structure and contents. This is useful for the LLM to understand the structure of a codebase, and to be able to reference specific files or directories.

For example:

- code_hierarchy
  - \_SignatureCaptureType
  - \_SignatureCaptureOptions
  - \_ScopeMethod
  - \_CommentOptions
  - \_ScopeItem
  - \_ChunkNodeOutput
  - CodeHierarchyNodeParser
    - class_name
    - **init**
    - \_get_node_name
      - recur
    - \_get_node_signature
      - find_start
      - find_end
    - \_chunk_node
    - get_code_hierarchy_from_nodes
      - get_subdict
      - recur_inclusive_scope
      - dict_to_markdown
    - \_parse_nodes
    - \_get_indentation
    - \_get_comment_text
    - \_create_comment_line
    - \_get_replacement_text
    - \_skeletonize
    - \_skeletonize_list
      - recur

## Usage as a Tool with an Agent

You can create a tool for any agent using the nodes from the node parser:

```python
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.packs.code_hierarchy import CodeHierarchyKeywordQueryEngine

query_engine = CodeHierarchyKeywordQueryEngine(
    nodes=split_nodes,
)

tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="code_lookup",
    description="Useful for looking up information about the code hierarchy codebase.",
)

agent = OpenAIAgent.from_tools(
    [tool], system_prompt=query_engine.get_tool_instructions(), verbose=True
)
```

## Adding new languages

To add a new language you need to edit `_DEFAULT_SIGNATURE_IDENTIFIERS` in `code_hierarchy.py`.

The docstrings are infomative as how you ought to do this and its nuances, it should work for most languages.

Please **test your new language** by adding a new file to `tests/file/code/` and testing all your edge cases.

People often ask "how do I find the Node Types I need for a new language?" The best way is to use breakpoints.
I have added a comment `TIP: This is a wonderful place to put a debug breakpoint` in the `code_hierarchy.py` file, put a breakpoint there, input some code in the desired language, and step through it to find the name
of the node you want to capture.

The code as it is should handle any language which:

1. expects you to indent deeper scopes
2. has a way to comment, either full line or between delimiters

## Future

I'm considering adding all the languages from [aider](https://github.com/paul-gauthier/aider/tree/main/aider/queries)
by incorporating `.scm` files instead of `_SignatureCaptureType`, `_SignatureCaptureOptions`, and `_DEFAULT_SIGNATURE_IDENTIFIERS`

## Contributing

You will need to set your `OPENAI_API_KEY` in your env to run the notebook or test the pack.

You can run tests with `pytest tests` in the root directory of this pack.
