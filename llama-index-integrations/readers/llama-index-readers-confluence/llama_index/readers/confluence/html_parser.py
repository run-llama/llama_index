class HtmlTextParser:
    def __init__(self):
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

        return markdownify(
            html,
            heading_style="ATX",  # Use # for headings instead of underlines
            bullets="*",  # Use * for unordered lists
            strip=["script", "style"],  # Remove script and style tags for security
        )
