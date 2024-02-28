import requests
import pandas as pd
import re

from typing import Optional


def request(url: str, method: str = "get", timeout: int = 10, **kwargs):
    """Helper to make requests from a url."""
    method = method.lower()
    assert method in [
        "delete",
        "get",
        "head",
        "patch",
        "post",
        "put",
    ], "Invalid request method."

    headers = kwargs.pop("headers", {})
    func = getattr(requests, method)
    return func(url, headers=headers, timeout=timeout, **kwargs)


def get_df(url: str, header: Optional[int] = None) -> pd.DataFrame:
    html = request(url).text
    # use regex to replace radio button html entries.
    html_clean = re.sub(r"(<span class=\"Fz\(0\)\">).*?(</span>)", "", html)
    return pd.read_html(html_clean, header=header)
