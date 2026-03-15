from bs4 import BeautifulSoup


def clean_spans(html: str) -> str:
    r"""
    Unwrap all <span> elements since they are semantically neutral.

    Spans cause markdownify to insert unwanted newlines (e.g. 'c\no\nn\ns\nt'
    instead of 'const'). Since span attributes have no meaning in markdown,
    all spans are unconditionally unwrapped.

    Args:
        html: Raw HTML string from Confluence

    Returns:
        Cleaned HTML with all spans unwrapped

    """
    if not html:
        return html

    soup = BeautifulSoup(html, "html.parser")

    for span in soup.find_all("span"):
        span.unwrap()

    return str(soup)


class HtmlTextParser:
    def __init__(self) -> None:
        try:
            from markdownify import markdownify  # noqa: F401
        except ImportError:
            raise ImportError(
                "`markdownify` package not found, please run `pip install markdownify`"
            )

    def convert(self, html: str) -> str:
        from markdownify import markdownify

        if not html:
            return ""

        cleaned_html = clean_spans(html)

        return markdownify(
            cleaned_html,
            heading_style="ATX",  # Use # for headings instead of underlines
            bullets="*",  # Use * for unordered lists
            strip=["script", "style"],  # Remove script and style tags for security
        )
