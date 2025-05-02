# Intro

Very often you have a large code base, with a rich docstrings and comments, that you would like to use to produce documentation. In fact, many open-source libraries like Scikit-learn or PyTorch have docstring so rich, that they contain LaTeX equations, or detailed examples.

At the same time, sometimes LLMs are used to read the full code from a repository, which can cost you many tokens, time and computational power.

DocstringWalker tries to find a sweet spot between these two approaches. You can use it to:

1. Parse all docstrings from modules, classes, and functions in your local code directory.
2. Convert them do LlamaIndex Documents.
3. Feed into LLM of your choice to produce a code-buddy chatbot or generate documentation.
   DocstringWalker utilizes only AST module, to process the code.

**With this tool, you can analyze only docstrings from the code, without the need to use tokens for the code itself.**

# Usage

Simply create a DocstringWalker and point it to the directory with the code. The class takes the following parameters:

1. Ignore **init**.py files - should **init**.py files be skipped? In some projects, they are not used at all, while in others they contain valuable info.
2. Fail on error - AST will throw SyntaxError when parsing a malformed file. Should this raise an exception for the whole process, or be ignored?

# Examples

Below you can find examples of using DocstringWalker.

## Example 1 - check Docstring Walker itself

Let's start by using it.... on itself :) We will see what information gets extracted from the module.

```python
# Step 1 - create docstring walker
walker = DocstringWalker()

# Step 2 - provide a path to... this directory :)
example1_docs = walker.load_data(docstring_walker_dir)

# Let's check docs content
print(example1_docs)

"""
[Document(id_=..., embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, hash=..., text="Module name: base \n Docstring: None...") ]
"""

# We can print the text of document
print(example1_docs[0].text[:500])

"""
Module name: base
Docstring: None
Class name: DocstringWalker
Docstring: A loader for docstring extraction and building structured documents from them.
Recursively walks a directory and extracts docstrings from each Python module - starting from the module
itself, then classes, then functions. Builds a graph of dependencies between the extracted docstrings.

Function name: load_data, In: DocstringWalker
Docstring: Load data from the specified code directory.
Additionally, after loading t
 """

# Step 3: Feed documents into Llama Index
example1_index = VectorStoreIndex(
    example1_docs, service_context=service_context
)

# Step 4: Query the index
example1_qe = example1_index.as_query_engine(service_context=service_context)


# Step 5: And start querying the index
print(
    example1_qe.query(
        "What are the main functions used by DocstringWalker? Describe each one in points."
    ).response
)

"""
1. load_data: This function loads data from a specified code directory and builds a dependency graph between the loaded documents. The graph is stored as an attribute of the class.

2. process_directory: This function processes a directory and extracts information from Python files. It returns a tuple containing a list of Document objects and a networkx Graph object. The Document objects represent the extracted information from Python files, and the Graph object represents the dependency graph between the extracted documents.

3. read_module_text: This function reads the text of a Python module given its path and returns the text of the module.

4. parse_module: This function parses a single Python module and returns a Document object with extracted information from the module.

5. process_class: This function processes a class node in the AST and adds relevant information to the graph. It returns a string representation of the processed class node and its sub-elements.

6. process_function: This function processes a function node in the AST and adds it to the graph. It returns a string representation of the processed function node with its sub-elements.

7. process_elem: This is a generic function that processes an element in the abstract syntax tree (AST) and delegates the execution to more specific functions based on the type of the element. It returns the result of processing the element.
"""
```

# Example 2 - check some arbitrarily selected module

Now we can check how to apply DocstringWalker to some files under an arbitrary directory. Let's use the code from the PyTorchGeometric KGE (Knowledge Graphs Embedding) directory.
You can find its original documentation and classes here: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#kge-models

We import the module and use its filepath directly.

```python
import os
from torch_geometric.nn import kge

# Step 1 - get path to module
module_path = os.path.dirname(kge.__file__)

# Step 2 - get the docs
example2_docs = walker.load_data(module_path)

# Step 3 - feed into Llama Index
example2_index = SummaryIndex.from_documents(
    example2_docs, service_context=service_context
)
example2_qe = example2_index.as_query_engine()

# Step 4 - query docstrings
print(
    example2_qe.query(
        "What classes are available and what is their main purpose? Use nested numbered list to describe: the class name, short summary of purpose, papers or literature review for each one of them."
    ).response
)


"""
1. DistMult
   - Purpose: Models relations as diagonal matrices, simplifying the bi-linear interaction between head and tail entities.
   - Paper: "Embedding Entities and Relations for Learning and Inference in Knowledge Bases" (https://arxiv.org/abs/1412.6575)

2. RotatE
   - Purpose: Models relations as a rotation in complex space from head to tail entities.
   - Paper: "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (https://arxiv.org/abs/1902.10197)

3. TransE
   - Purpose: Models relations as a translation from head to tail entities.
   - Paper: "Translating Embeddings for Modeling Multi-Relational Data" (https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)

4. KGEModel
   - Purpose: An abstract base class for implementing custom KGE models.

5. ComplEx
   - Purpose: Models relations as complex-valued bilinear mappings between head and tail entities using the Hermetian dot product.
   - Paper: "Complex Embeddings for Simple Link Prediction" (https://arxiv.org/abs/1606.06357)
"""
```
