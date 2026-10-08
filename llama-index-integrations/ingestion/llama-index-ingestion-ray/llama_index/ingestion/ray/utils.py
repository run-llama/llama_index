import pyarrow as pa
from llama_index.core.schema import BaseNode
from typing import Any, Sequence, List, Dict
import importlib

_ALLOWED_MODULE_PREFIXES = ("llama_index.",)


def _safe_load_node_class(module_name: str, class_name: str) -> type:
    # CWE-470: reject modules outside llama_index namespace
    if not any(module_name.startswith(p) for p in _ALLOWED_MODULE_PREFIXES):
        raise ValueError(
            f"Untrusted module '{module_name}': only llama_index.* modules are permitted."
        )
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    # Ensure the resolved class is actually a BaseNode subclass
    if not (isinstance(cls, type) and issubclass(cls, BaseNode)):
        raise ValueError(
            f"'{module_name}.{class_name}' is not a BaseNode subclass."
        )
    return cls


def ray_serialize_node(node: BaseNode) -> Dict[str, Any]:
    """Serialize a node to send to a Ray actor."""
    # Pop embedding to store separately/cleanly
    embedding = node.embedding
    node.embedding = None

    try:
        data = node.to_json()
    finally:
        # Always restore the embedding to avoid mutating the input object
        node.embedding = embedding

    return {
        "module": node.__class__.__module__,
        "class_name": node.__class__.__name__,
        "data": data,
        "embedding": embedding,
    }


def ray_serialize_node_batch(nodes: Sequence[BaseNode]) -> pa.Table:
    """Serialize a batch of nodes to send to a Ray actor."""
    modules = []
    class_names = []
    data_json = []
    embeddings = []

    for node in nodes:
        # 1. Capture embedding
        embed_val = node.embedding
        embeddings.append(embed_val)

        # 2. Pop embedding so it isn't included in the JSON string (save space)
        node.embedding = None

        # 3. Serialize remaining data to JSON
        try:
            modules.append(node.__class__.__module__)
            class_names.append(node.__class__.__name__)
            data_json.append(node.to_json())
        finally:
            # 4. Restore embedding so we don't destructively mutate the user's nodes
            node.embedding = embed_val

    return pa.Table.from_pydict(
        {
            "module": modules,
            "class_name": class_names,
            "data": data_json,  # This is now a column of JSON strings
            "embedding": embeddings,  # This is a column of float lists (or nulls)
        }
    )


def ray_deserialize_node(serialized_node: Dict[str, Any]) -> BaseNode:
    """Deserialize a node received from a Ray actor."""
    cls = _safe_load_node_class(serialized_node["module"], serialized_node["class_name"])

    # Reconstruct from JSON string
    node = cls.from_json(serialized_node["data"])

    # Re-attach embedding
    if serialized_node.get("embedding") is not None:
        node.embedding = serialized_node["embedding"]

    return node


def ray_deserialize_node_batch(table: pa.Table) -> List[BaseNode]:
    """Deserialize a batch of nodes received from a Ray actor."""
    return [ray_deserialize_node(row) for row in table.to_pylist()]
