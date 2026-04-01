from llama_index.tools.maxia import MaxiaToolSpec


def test_tool_spec_init() -> None:
    spec = MaxiaToolSpec()
    assert spec is not None


def test_spec_functions_count() -> None:
    assert len(MaxiaToolSpec.spec_functions) == 12


def test_to_tool_list() -> None:
    spec = MaxiaToolSpec()
    tools = spec.to_tool_list()
    assert len(tools) == 12
    names = {t.metadata.name for t in tools}
    assert "discover_services" in names
    assert "get_crypto_prices" in names
