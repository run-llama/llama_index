# **_Osmmap Loader_**

```bash
pip install llama-index-readers-maps
```

The Osmmap Loader will fetch map data from the [Overpass](https://wiki.openstreetmap.org/wiki/Main_Page) api for a certain place or area. Version **Overpass API 0.7.60** is used by this loader.

The api will provide you with all the **nodes, relations, and ways** for the particular region when you request data for a region or location.

## **Functions of the loader**

- To start, it first filters out those nodes that are already tagged, leaving just those nodes that are within 2 kilometres of the target location. The following keys are removed during filtering:["nodes," "geometry," "members"] from each node. The response we received is based on the tags and values we provided, so be sure to do that. The actions are covered below.

## **Steps to find the suitable tag and values**

1. Visit [Taginfo](taginfo.openstreetmap.org/tags). In essence, this website has all conceivable tags and values.
2. Perform a search for the feature you're looking for, for instance, "hospital" will return three results: "hospital" as an amenity, "hospital" as a structure, and "hospital" as a healthcare facility.
3. We may infer from the outcome that tag=amenity and value=hospital.
4. Leave the values parameter to their default value if you do not need to filter.

## **Usage**

The use case is here.

Let's meet **Jayasree**, who is extracting map features from her neighbourhood using the OSM map loader.
She requires all the nodes, routes, and relations within a five-kilometer radius of her locale (Guduvanchery).

- She must use the following arguments in order to accomplish the aforementioned. Localarea = "Guduvanchery" (the location she wants to seek), local_area_buffer = 5000 (5 km).

### And the code snippet looks like

```python
from llama_index.readers.maps import OpenMap

loader = MapReader()
documents = loader.load_data(
    localarea="Guduvanchery",
    search_tag="",
    tag_only=True,
    local_area_buffer=5000,
    tag_values=[""],
)
```

### Now she wants only the list hospitals around the location

- so she search for hospital tag in the [Taginfo](https://taginfo.openstreetmap.org/tags) and she got

```python
from llama_index.readers.maps import OpenMap

loader = MapReader()
documents = loader.load_data(
    localarea="Guduvanchery",
    search_tag="amenity",
    tag_only=True,
    local_area_buffer=5000,
    tag_values=["hospital", "clinic"],
)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
