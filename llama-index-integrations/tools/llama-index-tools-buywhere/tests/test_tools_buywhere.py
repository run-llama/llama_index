from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.buywhere import BuyWhereToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in BuyWhereToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    expected = {
        "search_products",
        "get_product",
        "compare_prices",
        "get_affiliate_link",
        "get_catalog",
    }
    assert set(BuyWhereToolSpec.spec_functions) == expected
