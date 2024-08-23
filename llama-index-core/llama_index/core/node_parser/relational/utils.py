from typing import Any

from io import StringIO


def md_to_df(md_str: str) -> Any:
    """Convert Markdown to dataframe."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "You must install the `pandas` package to use this node parser."
        )

    # Replace " by "" in md_str
    md_str = md_str.replace('"', '""')

    # Replace markdown pipe tables with commas
    md_str = md_str.replace("|", '","')

    # Remove the second line (table header separator)
    lines = md_str.split("\n")
    md_str = "\n".join(lines[:1] + lines[2:])

    # Remove the first and last second char of the line (the pipes, transformed to ",")
    lines = md_str.split("\n")
    md_str = "\n".join([line[2:-2] for line in lines])

    # Check if the table is empty
    if len(md_str) == 0:
        return None

    # Use pandas to read the CSV string into a DataFrame
    return pd.read_csv(StringIO(md_str))


def html_to_df(html_str: str) -> Any:
    """Convert HTML to dataframe."""
    try:
        from lxml import html
    except ImportError:
        raise ImportError(
            "You must install the `lxml` package to use this node parser."
        )

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "You must install the `pandas` package to use this node parser."
        )

    tree = html.fromstring(html_str)
    table_element = tree.xpath("//table")[0]
    rows = table_element.xpath(".//tr")

    data = []
    for row in rows:
        cols = row.xpath(".//td")
        cols = [c.text.strip() if c.text is not None else "" for c in cols]
        data.append(cols)

    # Check if the table is empty
    if len(data) == 0:
        return None

    # Check if the all rows have the same number of columns
    if not all(len(row) == len(data[0]) for row in data):
        return None

    return pd.DataFrame(data[1:], columns=data[0])
