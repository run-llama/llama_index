import pytest

from llama_index.core.node_parser.relational.utils import html_to_df

pytest.importorskip("lxml")
pytest.importorskip("pandas")


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("<a href='/country'>Germany</a>", "Germany"),
        ("Some <b>bold <i>nested</i></b> text", "Some bold nested text"),
        ("<span>Ada</span> and <span>Grace</span>", "Ada and Grace"),
        ("  <span>Fish &amp; chips</span>  ", "Fish & chips"),
        ("  plain text  ", "plain text"),
        ("", ""),
    ],
)
def test_html_to_df_preserves_cell_text(cell: str, expected: str) -> None:
    frame = html_to_df(
        "<table><tr><td>Name</td><td>Code</td></tr>"
        f"<tr><td>{cell}</td><td>+49</td></tr></table>"
    )

    assert frame is not None
    assert frame.columns.tolist() == ["Name", "Code"]
    assert frame.values.tolist() == [[expected, "+49"]]


def test_html_to_df_preserves_formatted_column_names() -> None:
    frame = html_to_df(
        "<table><tr><td><b>Name</b></td><td>Country <i>code</i></td></tr>"
        "<tr><td>Ada</td><td>+44</td></tr></table>"
    )

    assert frame is not None
    assert frame.columns.tolist() == ["Name", "Country code"]
    assert frame.values.tolist() == [["Ada", "+44"]]


def test_html_to_df_still_rejects_ragged_tables() -> None:
    assert (
        html_to_df(
            "<table><tr><td>Name</td><td>Code</td></tr><tr><td>Ada</td></tr></table>"
        )
        is None
    )


def test_html_to_df_empty_table() -> None:
    assert html_to_df("<table></table>") is None
