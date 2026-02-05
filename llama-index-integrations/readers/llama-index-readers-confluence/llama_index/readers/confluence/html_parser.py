from bs4 import BeautifulSoup


def clean_syntax_highlighting_spans(html: str) -> str:
    r"""
    Remove syntax highlighting spans that cause markdownify newline issues.

    Confluence wraps code in per-character <span> tags for syntax highlighting.
    Without preprocessing, markdownify produces 'c\no\nn\ns\nt' instead of 'const'.

    Args:
        html: Raw HTML string from Confluence

    Returns:
        Cleaned HTML with syntax highlighting spans unwrapped

    """
    if not html:
        return html

    soup = BeautifulSoup(html, "html.parser")

    for span in soup.find_all("span"):
        span_style = span.get("style", "")
        span_class = span.get("class", [])

        # Identify syntax highlighting spans by color/background styles or code-related classes
        is_syntax_span = (
            "color" in span_style
            or "background" in span_style
            or any("code" in c for c in span_class if isinstance(c, str))
        )

        if is_syntax_span:
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

        cleaned_html = clean_syntax_highlighting_spans(html)

        return markdownify(
            cleaned_html,
            heading_style="ATX",  # Use # for headings instead of underlines
            bullets="*",  # Use * for unordered lists
            strip=["script", "style"],  # Remove script and style tags for security
        )
