# Defining and Customizing Nodes

Nodes represent "chunks" of source Documents, whether that is a text chunk, an image, or more. They also contain metadata and relationship information
with other nodes and index structures.

Nodes are a first-class citizen in LlamaIndex. You can choose to define Nodes and all its attributes directly. You may also choose to "parse" source Documents into Nodes through our `NodeParser` classes.

For instance, you can do

```python
from llama_index.node_parser import SimpleNodeParser

parser = SimpleNodeParser.from_defaults()

nodes = parser.get_nodes_from_documents(documents)
```

You can also choose to construct Node objects manually and skip the first section. For instance,

```python
from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo

node1 = TextNode(text="<text_chunk>", id_="<node_id>")
node2 = TextNode(text="<text_chunk>", id_="<node_id>")
# set relationships
node1.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=node2.node_id)
node2.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=node1.node_id)
nodes = [node1, node2]
```

The `RelatedNodeInfo` class can also store additional `metadata` if needed:

```python
node2.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=node1.node_id, metadata={"key": "val"})
```

### Customizing the ID

Each node has an `node_id` property that is automatically generated if not manually specified. This ID can be used for 
a variety of purposes; this includes being able to update nodes in storage, being able to define relationships
between nodes (through `IndexNode`), and more.


You can also get and set the `node_id` of any `TextNode` directly (and also `Document` objects as well)!

```python
print(node.node_id)
node.node_id = "My new node_id!"

```


