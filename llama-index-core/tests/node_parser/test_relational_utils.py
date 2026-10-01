"""
`html_to_df` must read header cells written as `<th>`, not only `<td>`.

Most real HTML tables put their column names in `<th>` cells (often inside a
`<thead>`), which is exactly the shape `pd.read_html` and unstructured emit.
The cell query only matched `.//td`, so such a table produced an empty header
row while the body rows kept their cells. The "all rows have the same number
of columns" check then failed and the parser returned `None`, silently
dropping table-aware handling instead of returning the frame.
"""

import pytest

from llama_index.core.node_parser.relational.utils import html_to_df

try:
    from lxml import html  # noqa: F401  (presence check only)
except ImportError:
    html = None  # type: ignore

try:
    import pandas as pd  # noqa: F401  (presence check only)
except ImportError:
    pd = None  # type: ignore

lxml_required = pytest.mark.skipif(
    html is None or pd is None,
    reason="lxml and pandas must be installed to parse HTML tables",
)

TH_HEADER_TABLE = """
<table>
  <thead>
    <tr><th>Name</th><th>City</th></tr>
  </thead>
  <tbody>
    <tr><td>Ada</td><td>London</td></tr>
    <tr><td>Grace</td><td>New York</td></tr>
  </tbody>
</table>
"""

TD_HEADER_TABLE = """
<table>
  <tr><td>Name</td><td>City</td></tr>
  <tr><td>Ada</td><td>London</td></tr>
</table>
"""

TH_IN_BODY_TABLE = """
<table>
  <tr><th></th><th>Q1</th><th>Q2</th></tr>
  <tr><th>Revenue</th><td>1</td><td>2</td></tr>
</table>
"""


@lxml_required
def test_th_header_row_becomes_the_dataframe_columns() -> None:
    df = html_to_df(TH_HEADER_TABLE)

    assert df is not None, "a <th> header row was treated as an empty row"
    assert df.columns.tolist() == ["Name", "City"]
    assert df.values.tolist() == [["Ada", "London"], ["Grace", "New York"]]


@lxml_required
def test_row_headers_written_as_th_keep_their_cells() -> None:
    df = html_to_df(TH_IN_BODY_TABLE)

    assert df is not None, "a <th> header row left the table ragged"
    assert df.columns.tolist() == ["", "Q1", "Q2"]
    assert df.values.tolist() == [["Revenue", "1", "2"]]


@lxml_required
def test_td_only_tables_keep_parsing_the_same_way() -> None:
    df = html_to_df(TD_HEADER_TABLE)

    assert df is not None
    assert df.columns.tolist() == ["Name", "City"]
    assert df.values.tolist() == [["Ada", "London"]]


@lxml_required
def test_ragged_tables_are_still_rejected() -> None:
    ragged = """
    <table>
      <tr><th>Name</th><th>City</th></tr>
      <tr><td>Ada</td></tr>
    </table>
    """

    assert html_to_df(ragged) is None
