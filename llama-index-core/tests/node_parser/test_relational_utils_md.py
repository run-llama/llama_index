import pandas as pd
from pandas.testing import assert_frame_equal

from llama_index.core.node_parser.relational.utils import md_to_df


TABLE = "|Name|City|\n|---|---|\n|Ada|London|\n|Grace|New York|"


def test_md_to_df_crlf() -> None:
    expected = pd.DataFrame({"Name": ["Ada", "Grace"], "City": ["London", "New York"]})

    actual = md_to_df(TABLE.replace("\n", "\r\n"))

    assert actual is not None
    assert_frame_equal(actual, expected)


def test_md_to_df_trailing_whitespace() -> None:
    expected = pd.DataFrame({"Name": ["Ada", "Grace"], "City": ["London", "New York"]})
    for suffix in (" ", "  ", "\t", " \t "):
        table = "\n".join(line + suffix for line in TABLE.split("\n"))

        actual = md_to_df(table)

        assert actual is not None
        assert_frame_equal(actual, expected)


def test_md_to_df_crlf_with_trailing_whitespace() -> None:
    expected = pd.DataFrame({"Name": ["Ada", "Grace"], "City": ["London", "New York"]})
    table = "\r\n".join(line + "  " for line in TABLE.split("\n")) + "\r\n"

    actual = md_to_df(table)

    assert actual is not None
    assert_frame_equal(actual, expected)


def test_md_to_df_preserves_cell_content() -> None:
    table = '| Name | Description |\n| --- | --- |\n| Ada | A "quoted", value |'
    expected = pd.DataFrame(
        {" Name ": [" Ada "], " Description ": [' A "quoted", value ']}
    )

    actual = md_to_df(table)

    assert actual is not None
    assert_frame_equal(actual, expected)


def test_md_to_df_empty() -> None:
    assert md_to_df("") is None
