# LlamaIndex Usage Pattern

The general usage pattern of LlamaIndex is as follows:
1. Load in documents (either manually, or through a data loader)
2. Parse the Documents into Nodes
3. Construct Index (from Nodes or Documents)
4. [Optional, Advanced] Building indices on top of other indices
5. Query the index

## 1. Load in Documents

The first step is to load in data. This data is represented in the form of `Document` objects. 
We provide a variety of [data loaders](/how_to/data_connectors.md) which will load in Documents
through the `load_data` function, e.g.:

```python
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()

```

You can also choose to construct documents manually. LlamaIndex exposes the `Document` struct.

```python
from llama_index import Document

text_list = [text1, text2, ...]
documents = [Document(t) for t in text_list]
```

A Document represents a lightweight container around the data source. You can now to choose to proceed with one of the 
following steps:
1. Feed the Document object directly into the index (see section 3).
2. First convert the Document into Node objects (see section 2).

## 2. Parse the Documents into Nodes

The next step is to parse these Document objects into Node objects. Nodes represent "chunks" of source Documents,
whether that is a text chunk, an image, or more. They also contain metadata and relationship information
with other nodes and index structures.

Nodes are a first-class citizen in LlamaIndex. You can choose to define Nodes and all its attributes directly. You may also choose to "parse" source Documents into Nodes through our `NodeParser` classes.

For instance, you can do

```python
from llama_index.node_parser import SimpleNodeParser

parser = SimpleNodeParser()

nodes = parser.get_nodes_from_documents(documents)

```

You can also choose to construct Node objects manually and skip the first section. For instance,

```python
from llama_index.data_structs.node_v2 import Node, DocumentRelationship

node1 = Node(text="<text_chunk>", doc_id="<node_id>")
node2 = Node(text="<text_chunk>", doc_id="<node_id>")
# set relationships
node1.relationships[DocumentRelationship.NEXT] = node2.get_doc_id()
node2.relationships[DocumentRelationship.PREVIOUS] = node1.get_doc_id()

```


## 3. Index Construction

We can now build an index over these Document objects. The simplest high-level abstraction is to load-in the Document objects during index initialization (this is relevant if you came directly from step 1 and skipped step 2).

```python
from llama_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex.from_documents(documents)

```

You can also choose to build an index over a set of Node objects directly (this is a continuation of step 2).

```python
from llama_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex(nodes)

```


Depending on which index you use, LlamaIndex may make LLM calls in order to build the index.

### Inserting Documents

You can also take advantage of the `insert` capability of indices to insert Document objects
one at a time instead of during index construction. 

```python
from llama_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex([])
for doc in documents:
    index.insert(doc)

```

See the [Update Index How-To](/how_to/index_structs/update.md) for details and an example notebook.

**NOTE**: An `insert_node` function is coming!

### Customizing LLM's

By default, we use OpenAI's `text-davinci-003` model. You may choose to use another LLM when constructing
an index.

```python
from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext

...

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

index = GPTSimpleVectorIndex.from_documents(
    documents, service_context=service_context
)
```

See the [Custom LLM's How-To](/how_to/customization/custom_llms.md) for more details.


#### Customizing Prompts

Depending on the index used, we used default prompt templates for constructing the index (and also insertion/querying).
See [Custom Prompts How-To](/how_to/customization/custom_prompts.md) for more details on how to customize your prompt.


### Customizing embeddings

For embedding-based indices, you can choose to pass in a custom embedding model. See 
[Custom Embeddings How-To](custom-embeddings) for more details.


### Cost Predictor

Creating an index, inserting to an index, and querying an index may use tokens. We can track 
token usage through the outputs of these operations. When running operations, 
the token usage will be printed.
You can also fetch the token usage through `index.llm_predictor.last_token_usage`.
See [Cost Predictor How-To](/how_to/analysis/cost_analysis.md) for more details.


### [Optional] Save the index for future use

To save to disk and load from disk, do

```python
# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index.json')
```

## 4. [Optional, Advanced] Building indices on top of other indices

You can build indices on top of other indices! 

```python
from llama_index import GPTSimpleVectorIndex, GPTListIndex

index1 = GPTSimpleVectorIndex.from_documents(documents1)
index2 = GPTSimpleVectorIndex.from_documents(documents2)

# Set summary text
# you can set the summary manually, or you can
# generate the summary itself using LlamaIndex
index1.set_text("summary1")
index2.set_text("summary2")

index3 = GPTListIndex([index1, index2])

```

Composability gives you greater power in indexing your heterogeneous sources of data. For a discussion on relevant use cases,
see our [Query Use Cases](/use_cases/queries.md). For technical details and examples, see our [Composability How-To](/how_to/index_structs/composability.md).

## 5. Query the index.

After building the index, you can now query it. Note that a "query" is simply an input to an LLM - 
this means that you can use the index for question-answering, but you can also do more than that! 

To start, you can query an index without specifying any additional keyword arguments, as follows:

```python
response = index.query("What did the author do growing up?")
print(response)

response = index.query("Write an email to the user given their background information.")
print(response)
```

However, you also have a variety of keyword arguments at your disposal, depending on the type of
index being used. A full treatment of all the index-dependent query keyword arguments can be 
found [here](/reference/query.rst).

### Setting `mode`

An index can have a variety of query modes. For instance, you can choose to specify
`mode="default"` or `mode="embedding"` for a list index. `mode="default"` will a
create and refine an answer sequentially through the nodes of the list. 
`mode="embedding"` will synthesize an answer by fetching the top-k
nodes by embedding similarity.

```python
index = GPTListIndex.from_documents(documents)
# mode="default"
response = index.query("What did the author do growing up?", mode="default")
# mode="embedding"
response = index.query("What did the author do growing up?", mode="embedding")

```

The full set of modes per index are documented in the [Query Reference](/reference/query.rst).

(setting-response-mode)=
### Setting `response_mode`

Note: This option is not available/utilized in `GPTTreeIndex`.

An index can also have the following response modes through `response_mode`:
- `default`: For the given index, "create and refine" an answer by sequentially going through each Node; 
    make a separate LLM call per Node. Good for more detailed answers.
- `compact`: For the given index, "compact" the prompt during each LLM call by stuffing as 
    many Node text chunks that can fit within the maximum prompt size. If there are 
    too many chunks to stuff in one prompt, "create and refine" an answer by going through
    multiple prompts.
- `tree_summarize`: Given a set of Nodes and the query, recursively construct a tree 
    and return the root node as the response. Good for summarization purposes.

```python
index = GPTListIndex.from_documents(documents)
# mode="default"
response = index.query("What did the author do growing up?", response_mode="default")
# mode="compact"
response = index.query("What did the author do growing up?", response_mode="compact")
# mode="tree_summarize"
response = index.query("What did the author do growing up?", response_mode="tree_summarize")
```


### Setting `required_keywords` and `exclude_keywords`

You can set `required_keywords` and `exclude_keywords` on most of our indices (the only exclusion is the GPTTreeIndex). This will preemptively filter out nodes that do not contain `required_keywords` or contain `exclude_keywords`, reducing the search space
and hence time/number of LLM calls/cost.

```python
index.query(
    "What did the author do after Y Combinator?", required_keywords=["Combinator"], 
    exclude_keywords=["Italy"]
)
```



## 5. Parsing the response

The object returned is a [`Response` object](/reference/response.rst).
The object contains both the response text as well as the "sources" of the response:

```python
response = index.query("<query_str>")

# get response
# response.response
str(response)

# get sources
response.source_nodes
# formatted sources
response.get_formatted_sources()
```

An example is shown below. 
![](/_static/response/response_1.jpeg)



