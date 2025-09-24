import pytest

from llama_index.vector_stores.solr.types import BoostedTextField, SolrQueryDict


@pytest.mark.parametrize(
    ("field", "boost", "expected"),
    [
        ("title", 1.0, "title"),
        ("body", 2.5, "body^2.5"),
        ("summary", 0.8, "summary^0.8"),
        ("content", 0.0, "content^0.0"),
        ("abstract", 10.0, "abstract^10.0"),
    ],
    ids=["No boost", "Boost > 1", "Boost < 1", "Zero boost", "Large boost"],
)
def test_boosted_text_field_get_query_str(
    field: str, boost: float, expected: str
) -> None:
    field = BoostedTextField(field=field, boost_factor=boost)

    result = field.get_query_str()

    assert result == expected


def test_solr_query_dict_typing():
    # ensure we can construct a dict conforming to the TypedDict
    query: SolrQueryDict = {  # type: ignore[assignment]
        "q": "*:*",
        "fq": [],
    }
    # optional fields
    query["fl"] = "*"
    query["rows"] = "10"
    assert query["q"] == "*:*"
    assert query["fq"] == []
    assert query["fl"] == "*"
    assert query["rows"] == "10"
