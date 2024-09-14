# Using a Property Graph Index

A property graph is a knowledge collection of labeled nodes (i.e. entity categories, text labels, etc.) with properties (i.e. metadata), linked together by relationships into structured paths.

In LlamaIndex, the `PropertyGraphIndex` provides key orchestration around

- constructing a graph
- querying a graph

## Usage

Basic usage can be found by simply importing the class and using it:

```python
from llama_index.core import PropertyGraphIndex

# create
index = PropertyGraphIndex.from_documents(
    documents,
)

# use
retriever = index.as_retriever(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
)
nodes = retriever.retrieve("Test")

query_engine = index.as_query_engine(
    include_text=True,  # include source chunk with matching paths
    similarity_top_k=2,  # top k for vector kg node retrieval
)
response = query_engine.query("Test")

# save and load
index.storage_context.persist(persist_dir="./storage")

from llama_index.core import StorageContext, load_index_from_storage

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage")
)

# loading from existing graph store (and optional vector store)
# load from existing graph/vector store
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store, vector_store=vector_store, ...
)
```

### Construction

Property graph construction in LlamaIndex works by performing a series of `kg_extractors` on each chunk, and attaching entities and relations as metadata to each llama-index node. You can use as many as you like here, and they will all get applied.

If you've used transformations or metadata extractors with the [ingestion pipeline](../loading/ingestion_pipeline/index.md), then this will be very familiar (and these `kg_extractors` are compatible with the ingestion pipeline)!

Extractors are set using the appropriate kwarg:

```python
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[extractor1, extractor2, ...],
)

# insert additional documents / nodes
index.insert(document)
index.insert_nodes(nodes)
```

If not provided, the defaults are `SimpleLLMPathExtractor` and `ImplicitPathExtractor`.

All `kg_extractors` are detailed below.

#### (default) `SimpleLLMPathExtractor`

Extract short statements using an LLM to prompt and parse single-hop paths in the format (`entity1`, `relation`, `entity2`)

```python
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=10,
    num_workers=4,
    show_progress=False,
)
```

If you want, you can also customize the prompt and the function used to parse the paths.

Here's a simple (but naive) example:

```python
prompt = (
    "Some text is provided below. Given the text, extract up to "
    "{max_paths_per_chunk} "
    "knowledge triples in the form of `subject,predicate,object` on each line. Avoid stopwords.\n"
)


def parse_fn(response_str: str) -> List[Tuple[str, str, str]]:
    lines = response_str.split("\n")
    triples = [line.split(",") for line in lines]
    return triples


kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    extract_prompt=prompt,
    parse_fn=parse_fn,
)
```

#### (default) `ImplicitPathExtractor`

Extract paths using the `node.relationships` attribute on each llama-index node object.

This extractor does not need an LLM or embedding model to run, since it's merely parsing properties that already exist on llama-index node objects.

```python
from llama_index.core.indices.property_graph import ImplicitPathExtractor

kg_extractor = ImplicitPathExtractor()
```

### `DynamicLLMPathExtractor`

Will extract paths (including entity types!) according to optional list of allowed entity types and relation types. If none are provided, then the LLM will assign types as it sees fit. If they are provided, it will help guide the LLM, but will not enforce exactly those types.

```python
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

kg_extractor = DynamicLLMPathExtractor(
    llm=llm,
    max_triplets_per_chunk=20,
    num_workers=4,
    allowed_entity_types=["POLITICIAN", "POLITICAL_PARTY"],
    allowed_relation_types=["PRESIDENT_OF", "MEMBER_OF"],
)
```

#### `SchemaLLMPathExtractor`

Extract paths following a strict schema of allowed entities, relationships, and which entities can be connected to which relationships.

Using pydantic, structured outputs from LLMs, and some clever validation, we can dynamically specify a schema and verify the extractions per-path.

```python
from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

# recommended uppercase, underscore separated
entities = Literal["PERSON", "PLACE", "THING"]
relations = Literal["PART_OF", "HAS", "IS_A"]
schema = {
    "PERSON": ["PART_OF", "HAS", "IS_A"],
    "PLACE": ["PART_OF", "HAS"],
    "THING": ["IS_A"],
}

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=schema,
    strict=True,  # if false, will allow triples outside of the schema
    num_workers=4,
    max_paths_per_chunk=10,
    show_progres=False,
)
```

This extractor is extremely customizable, and has options to customize
- various aspects of the schema (as seen above)
- the `extract_prompt`
- `strict=False` vs. `strict=True`, to allow triples outside of the schema or not
- passing in your own custom `kg_schema_cls` if you are a pydantic pro and wanted to create you own pydantic class with custom validation.

### Retrieval and Querying

Labeled property graphs can be queried in several ways to retrieve nodes and paths. And in LlamaIndex, we can combine several node retrieval methods at once!

```python
# create a retriever
retriever = index.as_retriever(sub_retrievers=[retriever1, retriever2, ...])

# create a query engine
query_engine = index.as_query_engine(
    sub_retrievers=[retriever1, retriever2, ...]
)
```

If no sub-retrievers are provided, the defaults are
`LLMSynonymRetriever` and `VectorContextRetriever` (if embeddings are enabled).

All retrievers currently include:
- `LLMSynonymRetriever` - retrieve based on LLM generated keywords/synonyms
- `VectorContextRetriever` - retrieve based on embedded graph nodes
- `TextToCypherRetriever` - ask the LLM to generate cypher based on the schema of the property graph
- `CypherTemplateRetriever` - use a cypher template with params inferred by the LLM
- `CustomPGRetriever` - easy to subclass and implement custom retrieval logic

Generally, you would define one or more of these sub-retrievers and pass them to the `PGRetriever`:

```python
from llama_index.core.indices.property_graph import (
    PGRetriever,
    VectorContextRetriever,
    LLMSynonymRetriever,
)

sub_retrievers = [
    VectorContextRetriever(index.property_graph_store, ...),
    LLMSynonymRetriever(index.property_graph_store, ...),
]

retriever = PGRetriever(sub_retrievers=sub_retrievers)

nodes = retriever.retrieve("<query>")
```

Read on below for more details on all retrievers.

#### (default) `LLMSynonymRetriever`

The `LLMSynonymRetriever` takes the query, and tries to generate keywords and synonyms to retrieve nodes (and therefore the paths connected to those nodes).

Explicitly declaring the retriever allows you to customize several options. Here are the defaults:

```python
from llama_index.core.indices.property_graph import LLMSynonymRetriever

prompt = (
    "Given some initial query, generate synonyms or related keywords up to {max_keywords} in total, "
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...'\n"
    "Note, result should be in one-line, separated by '^' symbols."
    "----\n"
    "QUERY: {query_str}\n"
    "----\n"
    "KEYWORDS: "
)


def parse_fn(self, output: str) -> list[str]:
    matches = output.strip().split("^")

    # capitalize to normalize with ingestion
    return [x.strip().capitalize() for x in matches if x.strip()]


synonym_retriever = LLMSynonymRetriever(
    index.property_graph_store,
    llm=llm,
    # include source chunk text with retrieved paths
    include_text=False,
    synonym_prompt=prompt,
    output_parsing_fn=parse_fn,
    max_keywords=10,
    # the depth of relations to follow after node retrieval
    path_depth=1,
)

retriever = index.as_retriever(sub_retrievers=[synonym_retriever])
```

#### (default, if supported) `VectorContextRetriever`

The `VectorContextRetriever` retrieves nodes based on their vector similarity, and then fetches the paths connected to those nodes.

If your graph store supports vectors, then you only need to manage that graph store for storage. Otherwise, you will need to provide a vector store in addition to the graph store (by default, uses the in-memory `SimpleVectorStore`).

```python
from llama_index.core.indices.property_graph import VectorContextRetriever

vector_retriever = VectorContextRetriever(
    index.property_graph_store,
    # only needed when the graph store doesn't support vector queries
    # vector_store=index.vector_store,
    embed_model=embed_model,
    # include source chunk text with retrieved paths
    include_text=False,
    # the number of nodes to fetch
    similarity_top_k=2,
    # the depth of relations to follow after node retrieval
    path_depth=1,
    # can provide any other kwargs for the VectorStoreQuery class
    ...,
)

retriever = index.as_retriever(sub_retrievers=[vector_retriever])
```

#### `TextToCypherRetriever`

The `TextToCypherRetriever` uses a graph store schema, your query, and a prompt template for text-to-cypher in order to generate and execute a cypher query.

**NOTE:** Since the `SimplePropertyGraphStore` is not actually a graph database, it does not support cypher queries.

You can inspect the schema by using `index.property_graph_store.get_schema_str()`.

```python
from llama_index.core.indices.property_graph import TextToCypherRetriever

DEFAULT_RESPONSE_TEMPLATE = (
    "Generated Cypher query:\n{query}\n\n" "Cypher Response:\n{response}"
)
DEFAULT_ALLOWED_FIELDS = ["text", "label", "type"]

DEFAULT_TEXT_TO_CYPHER_TEMPLATE = (
    index.property_graph_store.text_to_cypher_template,
)


cypher_retriever = TextToCypherRetriever(
    index.property_graph_store,
    # customize the LLM, defaults to Settings.llm
    llm=llm,
    # customize the text-to-cypher template.
    # Requires `schema` and `question` template args
    text_to_cypher_template=DEFAULT_TEXT_TO_CYPHER_TEMPLATE,
    # customize how the cypher result is inserted into
    # a text node. Requires `query` and `response` template args
    response_template=DEFAULT_RESPONSE_TEMPLATE,
    # an optional callable that can clean/verify generated cypher
    cypher_validator=None,
    # allowed fields in the resulting
    allowed_output_field=DEFAULT_ALLOWED_FIELDS,
)
```

**NOTE:** Executing arbitrary cypher has its risks. Ensure you take the needed measures (read-only roles, sandboxed env, etc.) to ensure safe usage in a production environment.

#### `CypherTemplateRetriever`

This is a more constrained version of the `TextToCypherRetriever`. Rather than letting the LLM have free-range of generating any cypher statement, we can instead provide a cypher template and have the LLM fill in the blanks.

To illustrate how this works, here is a small example:

```python
# NOTE: current v1 is needed
from pydantic import BaseModel, Field
from llama_index.core.indices.property_graph import CypherTemplateRetriever

# write a query with template params
cypher_query = """
MATCH (c:Chunk)-[:MENTIONS]->(o)
WHERE o.name IN $names
RETURN c.text, o.name, o.label;
"""


# create a pydantic class to represent the params for our query
# the class fields are directly used as params for running the cypher query
class TemplateParams(BaseModel):
    """Template params for a cypher query."""

    names: list[str] = Field(
        description="A list of entity names or keywords to use for lookup in a knowledge graph."
    )


template_retriever = CypherTemplateRetriever(
    index.property_graph_store, TemplateParams, cypher_query
)
```

## Storage

Currently, supported graph stores for property graphs include:

|                              | In-Memory  | Native Embedding Support | Async | Server or disk based? |
|------------------------------|------------|--------------------------|-------|-----------------------|
| SimplePropertyGraphStore     | ✅         | ❌                       | ❌    | Disk                  |
| Neo4jPropertyGraphStore      | ❌         | ✅                       | ❌    | Server                |
| NebulaPropertyGraphStore     | ❌         | ❌                       | ❌    | Server                |
| TiDBPropertyGraphStore       | ❌         | ✅                       | ❌    | Server                |
| FalkorDBPropertyGraphStore   | ❌         | ✅                       | ❌    | Server                |

### Saving to/from disk

The default property graph store, `SimplePropertyGraphStore`, stores everything in memory and persists and loads from disk.

Here's an example of saving/loading an index with the default graph store:

```python
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices import PropertyGraphIndex

# create
index = PropertyGraphIndex.from_documents(documents)

# save
index.storage_context.persist("./storage")

# load
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### Saving and Loading with Integrations

Integrations typically save automatically. Some graph stores will support vectors, others might not. You can always combine a graph store with an external vector db as well.

This example shows how you might save/load a property graph index using Neo4j and Qdrant.

**Note:** If qdrant wasn't passed in, neo4j would store and use the embeddings on its own. This example illustrates the flexibility beyond that.

`pip install llama-index-graph-stores-neo4j llama-index-vector-stores-qdrant`

```python
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient

vector_store = QdrantVectorStore(
    "graph_collection",
    client=QdrantClient(...),
    aclient=AsyncQdrantClient(...),
)

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="<password>",
    url="bolt://localhost:7687",
)

# creates an index
index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    # optional, neo4j also supports vectors directly
    vector_store=vector_store,
    embed_kg_nodes=True,
)

# load from existing graph/vector store
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    # optional, neo4j also supports vectors directly
    vector_store=vector_store,
    embed_kg_nodes=True,
)
```

### Using the Property Graph Store Directly

The base storage class for property graphs is the `PropertyGraphStore`. These property graph stores are constructured using different types of `LabeledNode` objects, and connected using `Relation` objects.

We can create these ourselves, and also insert ourselves!

```python
from llama_index.core.graph_stores import (
    SimplePropertyGraphStore,
    EntityNode,
    Relation,
)
from llama_index.core.schema import TextNode

graph_store = SimplePropertyGraphStore()

entities = [
    EntityNode(name="llama", label="ANIMAL", properties={"key": "val"}),
    EntityNode(name="index", label="THING", properties={"key": "val"}),
]

relations = [
    Relation(
        label="HAS",
        source_id=entities[0].id,
        target_id=entities[1].id,
        properties={},
    )
]

graph_store.upsert_nodes(entities)
graph_store.upsert_relations(relations)

# optionally, we can also insert text chunks
source_chunk = TextNode(id_="source", text="My llama has an index.")

# create relation for each of our entities
source_relations = [
    Relation(
        label="HAS_SOURCE",
        source_id=entities[0].id,
        target_id="source",
    ),
    Relation(
        label="HAS_SOURCE",
        source_id=entities[1].id,
        target_id="source",
    ),
]
graph_store.upsert_llama_nodes([source_chunk])
graph_store.upsert_relations(source_relations)
```

Other helpful methods on the graph store include:
- `graph_store.get(ids=[])` - gets nodes based on ids
- `graph_store.get(properties={"key": "val"})` - gets nodes based on matching properties
- `graph_store.get_rel_map([entity_node], depth=2)` - gets triples up to a certain depth
- `graph_store.get_llama_nodes(['id1'])` - gets the original text nodes
- `graph_store.delete(ids=['id1'])` - delete based on ids
- `graph_store.delete(properties={"key": "val"})` - delete based on properties
- `graph_store.structured_query("<cypher query>")` - runs a cypher query (assuming the graph store supports it)

In addition `a` versions exist for all of these for async support (i.e. `aget`, `adelete`, etc.).

## Advanced Customization

As with all components in LlamaIndex, you can sub-class modules and customize things to work exactly as you need, or try out new ideas and research new modules!

### Sub-Classing Extractors

Graph extractors in LlamaIndex subclass the `TransformComponent` class. If you've worked with the ingestion pipeline before, this will be familiar since it is the same class.

The requirement for extractors is that the insert graph data into the metadata of the node, which will then be processed later on by the index.

Here is a small example of sub-classing to create a custom extractor:

```python
from llama_index.core.graph_store.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.schema import BaseNode, TransformComponent


class MyGraphExtractor(TransformComponent):
    # the init is optional
    # def __init__(self, ...):
    #     ...

    def __call__(
        self, llama_nodes: list[BaseNode], **kwargs
    ) -> list[BaseNode]:
        for llama_node in llama_nodes:
            # be sure to not overwrite existing entities/relations

            existing_nodes = llama_node.metadata.pop(KG_NODES_KEY, [])
            existing_relations = llama_node.metadata.pop(KG_RELATIONS_KEY, [])

            existing_nodes.append(
                EntityNode(
                    name="llama", label="ANIMAL", properties={"key": "val"}
                )
            )
            existing_nodes.append(
                EntityNode(
                    name="index", label="THING", properties={"key": "val"}
                )
            )

            existing_relations.append(
                Relation(
                    label="HAS",
                    source_id="llama",
                    target_id="index",
                    properties={},
                )
            )

            # add back to the metadata

            llama_node.metadata[KG_NODES_KEY] = existing_nodes
            llama_node.metadata[KG_RELATIONS_KEY] = existing_relations

        return llama_nodes

    # optional async method
    # async def acall(self, llama_nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
    #    ...
```

### Sub-Classing Retrievers

The retriever is a bit more complicated than the extractors, and has it's own special class to help make sub-classing easier.

The return type of the retrieval is extremely flexible. It could be
- a string
- a `TextNode`
- a `NodeWithScore`
- a list of one of the above

Here is a small example of sub-classing to create a custom retriever:

```python
from llama_index.core.indices.property_graph import (
    CustomPGRetriever,
    CUSTOM_RETRIEVE_TYPE,
)


class MyCustomRetriever(CustomPGRetriever):
    def init(self, my_option_1: bool = False, **kwargs) -> None:
        """Uses any kwargs passed in from class constructor."""
        self.my_option_1 = my_option_1
        # optionally do something with self.graph_store

    def custom_retrieve(self, query_str: str) -> CUSTOM_RETRIEVE_TYPE:
        # some some operation with self.graph_store
        return "result"

    # optional async method
    # async def acustom_retrieve(self, query_str: str) -> str:
    #     ...


custom_retriever = MyCustomRetriever(graph_store, my_option_1=True)

retriever = index.as_retriever(sub_retrievers=[custom_retriever])
```

For more complicated customization and use-cases, it is recommended to check out the source code and directly sub-class `BasePGRetriever`.

# Examples

Below, you can find some example notebooks showcasing the `PropertyGraphIndex`

- [Basic Usage](../../examples/property_graph/property_graph_basic.ipynb)
- [Using Neo4j](../../examples/property_graph/property_graph_neo4j.ipynb)
- [Using Nebula](../../examples/property_graph/property_graph_nebula.ipynb)
- [Advanced Usage with Neo4j and local models](../../examples/property_graph/property_graph_advanced.ipynb)
- [Using a Property Graph Store](../../examples/property_graph/graph_store.ipynb)
- [Creating a Custom Graph Retriever](../../examples/property_graph/property_graph_custom_retriever.ipynb)
- [Comparing KG Extractors](../../examples/property_graph/Dynamic_KG_Extraction.ipynb)
