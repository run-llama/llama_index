"""Regression tests for the DynamicLLMPathExtractor JSON-fallback parsers.

The two ``default_parse_dynamic_triplets*`` helpers fall back to a regex when
``json.loads`` fails. The original regexes used long chains of ``(.*?)`` lazy
groups between delimiter classes that admitted the same characters the groups
could consume, which produced catastrophic backtracking on attacker-influenced
or naturally-malformed LLM output (huntr report
https://huntr.com/bounties/3e4d9d96-a6d1-42aa-a2cc-aea2059f7c0d).

These tests both pin the linear-time behaviour and exercise the happy paths so
the fallback still recovers triplets from minor JSON drift.
"""

import time

from llama_index.core.indices.property_graph.transformations.dynamic_llm import (
    default_parse_dynamic_triplets,
    default_parse_dynamic_triplets_with_props,
)


# Generous timeouts so this stays stable on slow CI runners; the vulnerable
# regexes spent tens of seconds to many minutes on these same payloads.
_REDOS_TIMEOUT_S = 2.0


def test_default_parse_dynamic_triplets_recovers_from_minor_json_drift() -> None:
    # Single-quoted keys would make json.loads fail and exercise the regex path.
    llm_output = (
        "[{'head': 'Paris', 'head_type': 'City', 'relation': 'capital_of',"
        " 'tail': 'France', 'tail_type': 'Country'}]"
    )
    triplets = default_parse_dynamic_triplets(llm_output)
    assert len(triplets) == 1
    head, rel, tail = triplets[0]
    assert head.name == "Paris"
    assert head.label == "City"
    assert rel.label == "capital_of"
    assert tail.name == "France"
    assert tail.label == "Country"


def test_default_parse_dynamic_triplets_redos_payload_is_fast() -> None:
    # Polynomial-blowup payload from the huntr PoC (sink #1). Pre-fix this
    # took ~30s for 8 KB and ~8 minutes for 17 KB.
    payload = (
        '"head":"a","head_type":"b","relation":"r",' * 400
    ) + ("c" * 400)

    start = time.perf_counter()
    triplets = default_parse_dynamic_triplets(payload)
    elapsed = time.perf_counter() - start

    assert elapsed < _REDOS_TIMEOUT_S, (
        f"default_parse_dynamic_triplets took {elapsed:.2f}s on a {len(payload)}B "
        "fallback input (suspected ReDoS regression)"
    )
    # The payload is intentionally incomplete (no tail/tail_type) so no triplets
    # should be produced; the important guarantee is that we return promptly.
    assert triplets == []


def test_default_parse_dynamic_triplets_with_props_recovers_from_minor_json_drift() -> None:
    llm_output = (
        "[{'head': 'Paris', 'head_type': 'City', 'head_props': {\"pop\": \"2M\"},"
        " 'relation': 'capital_of', 'relation_props': {\"since\": \"987\"},"
        " 'tail': 'France', 'tail_type': 'Country', 'tail_props': {\"area\": \"643801\"}}]"
    )
    triplets = default_parse_dynamic_triplets_with_props(llm_output)
    assert len(triplets) == 1
    head, rel, tail = triplets[0]
    assert head.name == "Paris"
    assert head.properties.get("pop") == "2M"
    assert rel.label == "capital_of"
    assert rel.properties.get("since") == "987"
    assert tail.properties.get("area") == "643801"


def test_default_parse_dynamic_triplets_with_props_redos_payload_is_fast() -> None:
    # Exponential-blowup payload from the huntr PoC (sink #2). Pre-fix this
    # took ~22s for ~4.7 KB and roughly doubled per +10 repetitions.
    payload = (
        '"head":"a","head_type":"b","head_props":{},"relation":"r",' * 80
    ) + ("c" * 80)

    start = time.perf_counter()
    triplets = default_parse_dynamic_triplets_with_props(payload)
    elapsed = time.perf_counter() - start

    assert elapsed < _REDOS_TIMEOUT_S, (
        f"default_parse_dynamic_triplets_with_props took {elapsed:.2f}s on a "
        f"{len(payload)}B fallback input (suspected ReDoS regression)"
    )
    assert triplets == []


def test_oversized_fallback_input_is_rejected() -> None:
    # Defense-in-depth: anything past the fallback length cap returns [] without
    # touching the regex engine at all.
    payload = "x" * 200_000
    start = time.perf_counter()
    assert default_parse_dynamic_triplets(payload) == []
    assert default_parse_dynamic_triplets_with_props(payload) == []
    elapsed = time.perf_counter() - start
    assert elapsed < _REDOS_TIMEOUT_S
