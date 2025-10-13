# LlamaIndex Graph Stores Integration: ApertureDB

[ApertureDB](https://www.aperturedata.io/) is a Multimodal Database.
The storage is modelled as a graph.

It can be used off [cloud](https://cloud.aperturedata.io/), [On premise](https://docs.aperturedata.io/Setup/server/Custom#customer-hosted-setup), and it comes with a public [community edition](https://docs.aperturedata.io/Setup/server/Local) which can run on a laptop too.

This integration implements the PropertyGraph interface of llama_index, which can be used to Store and query a Knowledge Graph using ApertureDB as the store.

Assuming a working and accessible instance of ApertureDB the following examples would work for adding nodes to your graph, and retrieving them.

```python
from llama_index.core.graph_stores.types import Relation, EntityNode
from llama_index.graph_stores.ApertureDB import ApertureDBGraphStore

entities = [
    EntityNode(label="PERSON", name="James"),
    EntityNode(label="DISH", name="Butter Chicken"),
    EntityNode(label="DISH", name="Scrambled Eggs"),
    EntityNode(label="INGREDIENT", name="Butter"),
    EntityNode(label="INGREDIENT", name="Chicken"),
    EntityNode(label="INGREDIENT", name="Eggs"),
    EntityNode(label="INGREDIENT", name="Salt"),
]

relations = [
    Relation(
        label="EATS",
        source_id=entities[0].id,
        target_id=entities[1].id,
    ),
    Relation(
        label="EATS",
        source_id=entities[0].id,
        target_id=entities[2].id,
    ),
    Relation(
        label="CONTAINS",
        source_id=entities[1].id,
        target_id=entities[3].id,
    ),
    Relation(
        label="HAS",
        source_id=entities[1].id,
        target_id=entities[4].id,
    ),
    Relation(
        label="COMPRISED_OF",
        source_id=entities[2].id,
        target_id=entities[5].id,
    ),
    Relation(
        label="GOT",
        source_id=entities[2].id,
        target_id=entities[6].id,
    ),
]
graph_store = ApertureDBGraphStore()
graph_store.upsert_nodes(entities)
graph_store.upsert_relations(relations)
```

Retrieve nodes:

```python
# get all.
print(pg_store.get())

# get nodes by ID.
kg_nodes = pg_store.get(ids=[entities[0].id])
print(kg_nodes)

# get paths from a node
paths = pg_store.get_rel_map(kg_nodes, depth=2)
import json

print(json.dumps(paths, indent=2, default=str))
```
