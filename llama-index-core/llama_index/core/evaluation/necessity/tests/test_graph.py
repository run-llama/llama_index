import pytest
from hypothesis import given, strategies as st
from hypothesis import settings, HealthCheck

from llama_index.core.evaluation.necessity.graph import (
    EvidenceNecessityGraph,
)
from llama_index.core.evaluation.necessity.necessity import (
    NecessityResult,
)


# ---------------------------------------------------------------------
# Core graph construction & queries
# ---------------------------------------------------------------------

def test_graph_add_and_query():
    """
    Graph must correctly store and return necessary contexts
    for a supported claim.
    """
    graph = EvidenceNecessityGraph()

    result = NecessityResult(
        claim="The Eiffel Tower is in Paris.",
        initially_supported=True,
        necessary_context_indices=[0],
    )

    graph.add_result(result)

    assert graph.necessary_contexts_for(
        "The Eiffel Tower is in Paris."
    ) == [0]

    assert graph.unsupported_claims() == []


def test_graph_unsupported_claim():
    """
    Unsupported claims must be tracked explicitly.
    """
    graph = EvidenceNecessityGraph()

    result = NecessityResult(
        claim="The Eiffel Tower is in Berlin.",
        initially_supported=False,
        necessary_context_indices=[],
    )

    graph.add_result(result)

    assert graph.unsupported_claims() == [
        "The Eiffel Tower is in Berlin."
    ]


def test_all_necessary_contexts():
    """
    Graph must return the union of all necessary context indices
    across claims, sorted and de-duplicated.
    """
    graph = EvidenceNecessityGraph()

    graph.add_result(
        NecessityResult(
            claim="Claim A",
            initially_supported=True,
            necessary_context_indices=[0, 1],
        )
    )

    graph.add_result(
        NecessityResult(
            claim="Claim B",
            initially_supported=True,
            necessary_context_indices=[1, 2],
        )
    )

    assert graph.all_necessary_contexts() == [0, 1, 2]


# ---------------------------------------------------------------------
# Redundancy & robustness analysis
# ---------------------------------------------------------------------

def test_fragile_and_robust_claims():
    """
    Fragile claims depend on exactly one context.
    Robust claims depend on multiple contexts.
    """
    graph = EvidenceNecessityGraph()

    graph.add_result(
        NecessityResult(
            claim="Claim A",
            initially_supported=True,
            necessary_context_indices=[0],
        )
    )

    graph.add_result(
        NecessityResult(
            claim="Claim B",
            initially_supported=True,
            necessary_context_indices=[1, 2],
        )
    )

    assert graph.fragile_claims() == ["Claim A"]
    assert graph.robust_claims() == ["Claim B"]


def test_redundant_contexts():
    """
    Redundant contexts are those not required by any claim.
    """
    graph = EvidenceNecessityGraph()

    graph.add_result(
        NecessityResult(
            claim="Claim A",
            initially_supported=True,
            necessary_context_indices=[0],
        )
    )

    graph.add_result(
        NecessityResult(
            claim="Claim B",
            initially_supported=True,
            necessary_context_indices=[2],
        )
    )

    # contexts: [0, 1, 2, 3]
    assert graph.redundant_contexts(
        total_contexts=4
    ) == [1, 3]


# ---------------------------------------------------------------------
# Serialization & schema guarantees
# ---------------------------------------------------------------------

def test_graph_round_trip_serialization():
    """
    Graph must support lossless round-trip serialization
    via to_dict() / from_dict().
    """
    graph = EvidenceNecessityGraph()

    graph.add_result(
        NecessityResult(
            claim="Claim A",
            initially_supported=True,
            necessary_context_indices=[0, 2],
        )
    )

    graph.add_result(
        NecessityResult(
            claim="Claim B",
            initially_supported=False,
            necessary_context_indices=[],
        )
    )

    payload = graph.to_dict()
    restored = EvidenceNecessityGraph.from_dict(payload)

    # Structural equivalence invariant
    assert restored.to_dict() == payload
    assert restored.unsupported_claims() == ["Claim B"]
    assert restored.all_necessary_contexts() == [0, 2]


def test_graph_schema_version_mismatch_raises():
    """
    Deserialization must fail fast on unsupported schema versions.
    """
    graph = EvidenceNecessityGraph()
    payload = graph.to_dict()

    # Simulate incompatible future version
    payload["version"] = payload["version"] + 1

    with pytest.raises(ValueError):
        EvidenceNecessityGraph.from_dict(payload)


# ---------------------------------------------------------------------
# Property-based invariants (randomized structure testing)
# ---------------------------------------------------------------------

@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    claims=st.lists(
        st.text(min_size=1),
        min_size=1,
        max_size=10,
        unique=True,
    ),
    context_indices=st.lists(
        st.integers(min_value=0, max_value=20),
        min_size=1,
        max_size=10,
    ),
)
def test_graph_property_invariants(claims, context_indices):
    """
    Property-based test:
    For any graph configuration, structural invariants must hold.
    """
    graph = EvidenceNecessityGraph()

    for claim in claims:
        graph.add_result(
            NecessityResult(
                claim=claim,
                initially_supported=True,
                necessary_context_indices=context_indices,
            )
        )

    # Invariant 1: all_necessary_contexts is sorted & unique
    all_ctx = graph.all_necessary_contexts()
    assert all_ctx == sorted(set(all_ctx))

    # Invariant 2: every necessary context appears in at least one edge
    for idx in all_ctx:
        assert any(
            idx in graph.necessary_contexts_for(c)
            for c in claims
        )

    # Invariant 3: no unsupported claims reported
    assert graph.unsupported_claims() == []


# ---------------------------------------------------------------------
# Defensive deserialization (fuzz testing)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        {},  # empty
        {"version": 1},  # missing claims
        {"claims": {}},  # missing version
        {"version": "1", "claims": {}},  # wrong type
        {"version": 1, "claims": {"C": {}}},  # missing fields
        {"version": 1, "claims": {"C": {"supported": True}}},  # missing indices
        {
            "version": 1,
            "claims": {
                "C": {
                    "supported": "yes",
                    "necessary_context_indices": [],
                }
            },
        },  # wrong supported type
    ],
)
def test_graph_from_dict_rejects_malformed_payloads(payload):
    """
    Fuzz test:
    Malformed payloads must fail fast and loudly.
    """
    with pytest.raises(Exception):
        EvidenceNecessityGraph.from_dict(payload)
