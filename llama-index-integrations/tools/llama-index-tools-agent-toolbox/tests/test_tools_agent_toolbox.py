from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.agent_toolbox import AgentToolboxToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in AgentToolboxToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    expected = [
        "search", "extract", "screenshot", "weather", "finance",
        "validate_email", "translate", "geoip", "news", "whois",
        "dns", "pdf_extract", "qr_generate",
    ]
    assert AgentToolboxToolSpec.spec_functions == expected


def test_init():
    spec = AgentToolboxToolSpec(api_key="atb_test_key")
    assert spec.api_key == "atb_test_key"
    assert spec.base_url == "https://api.sendtoclaw.com"


def test_to_tool_list():
    spec = AgentToolboxToolSpec(api_key="atb_test_key")
    tools = spec.to_tool_list()
    assert len(tools) == 13
    tool_names = [t.metadata.name for t in tools]
    assert "search" in tool_names
    assert "geoip" in tool_names
    assert "qr_generate" in tool_names
