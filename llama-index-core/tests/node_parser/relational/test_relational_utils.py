import pytest

from llama_index.core.node_parser.relational.utils import html_to_df

pd = pytest.importorskip("pandas")
pytest.importorskip("lxml")


def test_html_to_df_without_table_returns_none() -> None:
    """
    HTML that contains no <table> must return None, not raise IndexError.

    html_to_df is called on the ``text_as_html`` of parsed document elements,
    which is not guaranteed to wrap the content in a <table>. Accessing
    ``xpath("//table")[0]`` on such input crashed the whole node parser.
    """
    assert html_to_df("<div>Hello, no table here!</div>") is None


def test_html_to_df_parses_table() -> None:
    """A well-formed table is still parsed into a DataFrame."""
    result = html_to_df(
        "<table>"
        "<tr><td>name</td><td>age</td></tr>"
        "<tr><td>alice</td><td>30</td></tr>"
        "</table>"
    )
    assert result is not None
    assert list(result.columns) == ["name", "age"]
    assert result.shape == (1, 2)
    assert result.iloc[0]["name"] == "alice"
    assert result.iloc[0]["age"] == "30"
