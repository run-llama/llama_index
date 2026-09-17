import json

from llama_index.core.indices.property_graph.transformations.dynamic_llm import (
    default_parse_dynamic_triplets,
    default_parse_dynamic_triplets_with_props,
)


def test_parse_dynamic_triplets_accepts_object_json() -> None:
    raw = json.dumps(
        {
            "head": "Alice",
            "head_type": "Person",
            "relation": "KNOWS",
            "tail": "Bob",
            "tail_type": "Person",
        }
    )
    triplets = default_parse_dynamic_triplets(raw)
    assert len(triplets) == 1
    assert triplets[0][0].name == "Alice"
    assert triplets[0][2].name == "Bob"


def test_parse_dynamic_triplets_accepts_wrapped_list() -> None:
    raw = json.dumps(
        {
            "triplets": [
                {
                    "head": "Alice",
                    "head_type": "Person",
                    "relation": "KNOWS",
                    "tail": "Bob",
                    "tail_type": "Person",
                }
            ]
        }
    )
    triplets = default_parse_dynamic_triplets(raw)
    assert len(triplets) == 1
    assert triplets[0][1].label == "KNOWS"


def test_parse_dynamic_triplets_skips_non_dict_items() -> None:
    assert default_parse_dynamic_triplets('["not a triplet"]') == []


def test_parse_dynamic_triplets_with_props_accepts_object_json() -> None:
    raw = json.dumps(
        {
            "head": "Alice",
            "head_type": "Person",
            "head_props": {"age": 30},
            "relation": "KNOWS",
            "relation_props": {},
            "tail": "Bob",
            "tail_type": "Person",
            "tail_props": {},
        }
    )
    triplets = default_parse_dynamic_triplets_with_props(raw)
    assert len(triplets) == 1
    assert triplets[0][0].properties["age"] == 30
