from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.ictfax import ICTFaxToolSpec


def test_class():
    names = {b.__name__ for b in ICTFaxToolSpec.__mro__}
    assert BaseToolSpec.__name__ in names


def test_spec_functions():
    spec = ICTFaxToolSpec(base_url="http://x/api", username="u", password="p")
    assert set(spec.spec_functions) == {
        "send_fax", "get_fax_status", "list_faxes", "upload_fax_document",
    }
    # tool list builds without contacting the server
    assert len(spec.to_tool_list()) == 4
