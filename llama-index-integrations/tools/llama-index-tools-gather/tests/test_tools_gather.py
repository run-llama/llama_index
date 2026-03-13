from llama_index.tools.gather import GatherToolSpec


def test_class_exists():
    """Verify the tool spec can be instantiated."""
    spec = GatherToolSpec()
    assert spec is not None
    assert "gather_feed" in spec.spec_functions
    assert "gather_agents" in spec.spec_functions
    assert "gather_search" in spec.spec_functions
