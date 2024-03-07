# CodeHierarchyNodeParser

The `CodeHierarchyNodeParser` is useful to split long code files into more reasonable chunks. What this will do is create a "Hierarchy" of sorts, where sections of the code are made more reasonable by replacing the scope body with short comments telling the LLM to search for a referenced node if it wants to read that context body. This is called skeletonization, and is toggled by setting `skeleton` to `True` which it is by default.

Nodes in this hierarchy will be split based on scope, like function, class, or method scope, and will have links to their children and parents so the LLM can traverse the tree.

```python
from llama_index.core.text_splitter import CodeSplitter
from llama_index.core.llama_pack import download_llama_pack

CodeHierarchyNodeParser = download_llama_pack("CodeHierarchyNodeParser")

split_nodes = CodeHierarchyNodeParser(
    language="python",
    # You can further parameterize the CodeSplitter to split the code
    # into "chunks" that match your context window size using
    # chunck_lines and max_chars parameters, here we just use the defaults
    code_splitter=CodeSplitter(language="python"),
)
```

A full example can be found [here in combination with `CodeSplitter`](./CodeHierarchyNodeParserUsage.ipynb).

# Repo Maps

Generate a map of a repository's structure and contents. This is useful for the LLM to understand the structure of a codebase, and to be able to reference specific files or directories.

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

# Query Engine & Langchain Tool

Generates a langchain tool with the following name and description:

```
name: "Code Search"
description:
    Search the tool by any element in this list,
    or any uuid found in the code,
    to get more information about that element.

    {repo_map}
```

# Adding new languages

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

Please make a `.env` file in the root directory of this pack and then add your `OPENAI_API_KEY` to it to run the jupyter notebook.

You can run tests with `pytest tests` in the root directory of this pack.
